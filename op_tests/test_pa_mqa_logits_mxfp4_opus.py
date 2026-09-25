# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS) -- correctness and perf, through the arch dispatcher.

Tables: corner cases, causal prefill, CSA prefill (fresh, and chunked at PR #5332's shapes),
and MTP decode -- as markdown and as one-line JSON records for a benchmark driver. Every launch
goes through ``aiter.ops.opus.pa_mqa_logits_mxfp4``. Inputs use the natural (3-D) layouts on
gfx1250 and the MFMA-permuted (4-D) ones on gfx950; the reference always reads natural E8M0.

    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py             # the full default sweep
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py -b 1 2      # a quick subset
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py \\
        --data-init constant uniform --scale-init constant auto   # two paired init regimes

Data and scale are independent axes: the reference dequantizes any ``(nibbles, E8M0)`` pair, so
the exponent spread is set by ``--scale-init`` alone. The spread is what lets a misrouted scale
show: if rows 16 apart (one lane half) share an exponent, a wrong ``b_scale_sel`` reads a right
value. ``correctness_blindness`` skips the correctness sweep for an init pair that cannot fail.

The fp32 reference shares this file's view of the scale layout, so on gfx950 the corner cases
also get an independent FlyDSL cross-check built from the same natural bytes (``vs flydsl``).
"""

import argparse
import itertools
import math
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
from aiter.ops.opus._arch import GFX950, _device_arch
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
    _default_variant,
    _launch,
    pa_mqa_logits_mxfp4,
    pa_mqa_logits_mxfp4_block_table_width,
    pa_mqa_logits_mxfp4_plan,
    pa_mqa_logits_mxfp4_plan_buffers,
    pa_mqa_logits_mxfp4_variants,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

dev = "cuda"

SUPPORTED_GFX = ["gfx1250", "gfx950"]

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64  # page size
SCALE_BLOCK = 32  # E8M0 block
WEIGHT_SCALE = 1.5
BLOCKS_ROW = HEAD_DIM // SCALE_BLOCK  # 4 natural E8M0 blocks per row

# gfx950 MFMA-permuted scale geometry: a lane's 4 E8M0 bytes in one dword, [.., g, m, byte].
MFMA_N = 32
K_TILES = HEAD_DIM // 64  # 2  (MFMA_K = 64)
K_CHUNKS = 64 // SCALE_BLOCK  # 2  (32-K chunks per k-tile)
SCALE_BYTES = 4  # K_TILES * n_tiles per lane dword

CSA_RATIO = 4  # ATOM's compression ratio: row n sees floor((pos + 1) / 4)

# Decode windows are COMPRESSED column counts: 25000 at ratio 4 is ~100k raw tokens.
DECODE_SEQS = 32
DECODE_WIN_LONG = 25000
DECODE_WIN_SHORT = 100

# Window ends around the 64-token tile edges; 1-2 tile windows run the pipeline's peeled
# prologue and epilogue without a steady-state loop.
TILE_EDGE_ENDS = (1, 63, 64, 65, 127, 128, 129, 191, 255, 256, 257, 383, 384, 385)
PREFILL_TOTAL_QLEN = 16384
PREFILL_QMIN = 800
N_COS_SAMPLE = 8

# Pinned, not a flag: readings at different iteration counts are not comparable.
PERF_ITERS = 50
PERF_WARMUP = 10

# Per-row / per-page byte counts for the traffic denominator (the gfx950 layout only permutes).
Q_ROW_BYTES = HEADS * HEAD_DIM // 2  # 4096: one packed fp4 query row
QS_ROW_BYTES = HEADS * BLOCKS_ROW  # 256: its E8M0 scales, [H, 4]
W_ROW_BYTES = HEADS * 2  # 128: bf16 per-head weights
KV_PAGE_BYTES = KV_BLOCK_SIZE * HEAD_DIM // 2  # 4096: one packed fp4 page
KVS_PAGE_BYTES = KV_BLOCK_SIZE * BLOCKS_ROW  # 256: its E8M0 scales, [PAGE, 4]

# `fill_fp4`'s constant fills the packed byte and defaults to 0 (same as `zero`). 0x22 is two
# 1.0 nibbles, so an all-constant run puts every in-window cell at HEADS*HEAD_DIM*1.5 = 12288.
FP4_CONSTANT_BYTE = 0x22

# Minimum fraction of rows 16 apart with different exponents; below it the sweep is skipped.
MISROUTE_DISAGREE_MIN = 0.2

# fp32 reassociation bound, relative to the checked cells' max |ref|. A logit is a signed sum
# over 64 heads, so a cancelled cell still carries rounding from terms as large as the row's
# biggest logit (seen: 2.4e-7 of range). A misrouted scale moves a cell by 2x, far above this.
REASSOC_ATOL_REL = 1e-6


# ── the two init axes, and the dequant that joins them ────────────────────────
def fill_nibbles(rows, data_init, gen):
    """``[rows, HEAD_DIM/2]`` packed e2m1, low nibble = even element."""
    return fill_fp4(
        (rows, HEAD_DIM), data_init, gen, device=dev, constant=FP4_CONSTANT_BYTE
    )


def fill_exponents(shape, scale_init, gen):
    """E8M0 on-wire bytes at the library's default (widest) pow2 spread; a narrower one would
    blunt the misroute check, and ``REASSOC_ATOL_REL`` already covers the fp32 range."""
    return fill_scale_e8m0(shape, scale_init, gen, device=dev)


def fp4_dequant(packed, e8m0, block_size=SCALE_BLOCK):
    """``[..., d/2]`` packed e2m1 + ``[..., d/block]`` E8M0 -> ``[..., d]`` fp32.

    Defined for any pair. Both decodes come from ``aiter.utility.fp4_utils`` so the reference
    stays canonical, e.g. E8M0 ``0xFF`` decodes to NaN rather than +inf.
    """
    *prefix, d_half = packed.shape
    d = d_half * 2
    vals = mxfp4_to_f32(packed).reshape(*prefix, d // block_size, block_size)
    scale = e8m0_to_f32(e8m0)
    return (vals * scale.unsqueeze(-1)).reshape(*prefix, d)


# ── per-arch scale/cache layout ───────────────────────────────────────────────
def _is_permuted() -> bool:
    """True on gfx950 (MFMA-permuted layouts), False on gfx1250 (natural). Queried per call on
    the current device, never at import (the probe inits HIP before ``main()``'s arch gate)."""
    return _device_arch(torch.cuda.current_device()) == GFX950


def _scale_to_opus(e8_nat, rows_per_group):
    """``[rows, 4]`` natural E8M0 -> ``[rows/rpg, K_CHUNKS, MFMA_N, SCALE_BYTES]``, gfx950's
    layout; a pure permutation. ``rows_per_group``: 64 heads (q_scale) or 64 page tokens (kv).
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
    q_e8: torch.Tensor  # [T*H, 4] natural E8M0, the source of every q_scale layout
    kv_e8: torch.Tensor  # [nb*PAGE, 4] natural E8M0, the source of every kv_scale layout


def pages_for(max_end):
    """Pages per sequence, rounded up to a whole KV tile of the widest compiled variant, since a
    CTA reads ``block_tables`` for every page of its last tile. Uses the op's own sizing helper.
    """
    return pa_mqa_logits_mxfp4_block_table_width(
        max(max_end, 1), kv_block_size=KV_BLOCK_SIZE
    )


def build_inputs(bs, max_end, total_tokens, seed, data_init, scale_init):
    """Every buffer from one seeded generator; nibbles and exponents are drawn independently."""
    gen = make_generator(seed, device=dev)
    mbps = pages_for(max_end)
    t_max = mbps * KV_BLOCK_SIZE
    num_blocks = bs * mbps

    permuted = _is_permuted()

    # --- KV: the reference reads natural (kv_packed, kv_e8); the kernel the arch's layout. ---
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
        q_e8=q_e8,
        kv_e8=kv_e8,
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
    """Mismatch ratio over the sampled rows' in-window cells; ``atol`` scales with their range
    (see ``REASSOC_ATOL_REL``)."""
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
    """Lower bound on bytes moved: each KV page once per batch, not one K vector per logit
    (rows of a batch share pages, so per-logit counting exceeds HBM peak)."""
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
    """Every cell outside ``[local_start, local_end)`` must keep its -inf pre-fill; catches
    stray stores that leave the in-window values correct."""
    col = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    return bool(torch.isneginf(out[~inside]).all().item())


def window_is_written(out, ls, le):
    """Every cell inside ``[local_start, local_end)`` was stored, over all rows (``check_rows``
    only samples), so a dropped tail tile shows."""
    col = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    return bool(torch.isfinite(out[inside]).all().item())


def sample_rows(total, le, n=N_COS_SAMPLE, seed=0):
    nonempty = torch.nonzero(le > 0).flatten().tolist()
    if not nonempty:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(nonempty, min(n, len(nonempty))))


# ── gfx950 second opinion: FlyDSL, on its own layout ─────────────────────────
# FlyDSL's preshuffle is a different permutation (16-row MFMA tiles), built from the natural
# `q_e8` / `kv_e8`, so a layout bug shared by `_scale_to_opus` and the kernel cannot pass both.
FLY_MFMA_M = 16
FLY_KVS_NTPW = 4


def _q_scale_flydsl(e8_nat, total_tokens):
    """``[T*H, 4]`` -> FlyDSL's ``[T, D/128, 4, 16, QS_PAD]``."""
    m_tiles = HEADS // FLY_MFMA_M
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe = (
        e8_nat.reshape(total_tokens, m_tiles, FLY_MFMA_M, HEAD_DIM // 128, 4)
        .permute(0, 3, 4, 2, 1)  # [T, K_TILES_16, 4, 16, M_TILES]
        .contiguous()
    )
    return torch.nn.functional.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def _kv_scale_flydsl(e8_nat, num_blocks):
    """``[nb*PAGE, 4]`` -> FlyDSL's ``[nb, 1, 4, PAGE]``, token o at ``(o%16)*4 + o//16``."""
    o = torch.arange(KV_BLOCK_SIZE, device=e8_nat.device)
    sflat = (o % FLY_MFMA_M) * FLY_KVS_NTPW + (o // FLY_MFMA_M)
    out = torch.zeros(
        num_blocks, 1, 4, KV_BLOCK_SIZE, dtype=torch.uint8, device=e8_nat.device
    )
    out[:, 0, :, sflat] = e8_nat.reshape(num_blocks, KV_BLOCK_SIZE, 4).permute(0, 2, 1)
    return out.contiguous()


def flydsl_cross_check(inp, out, rb, ls, le, variant):
    """Mismatch ratio of ``out`` against FlyDSL over every in-window cell, or NaN when skipped:
    off gfx950, FlyDSL not importable, or any window start not a multiple of 4 (FlyDSL at
    ``num_warps = 1`` drops the leading unaligned cells).
    """
    if not _is_permuted() or bool((ls % 4 != 0).any().item()):
        return float("nan")
    try:
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
            flydsl_pa_mqa_logits_fp4_prefill,
        )
    except Exception as e:  # noqa: BLE001 -- any import failure means "no second opinion"
        aiter.logger.warning("FlyDSL cross-check unavailable: %s: %s", type(e).__name__, e)
        return float("nan")
    total_q = int(rb.numel())
    num_blocks = inp.block_tables.numel()
    msl = inp.max_seq_len
    _, cta, n_ctas = compute_prefill_schedule(rb, ls, le, variant.block_k, total_q, msl)
    fly = torch.full((total_q, msl), float("-inf"), dtype=torch.float32, device=dev)
    flydsl_pa_mqa_logits_fp4_prefill(
        inp.q_packed, _q_scale_flydsl(inp.q_e8, total_q),
        inp.kv_cache.view(-1, 1, 4, KV_BLOCK_SIZE, 16),
        _kv_scale_flydsl(inp.kv_e8, num_blocks), inp.block_tables, inp.weights,
        rb, ls, le, msl,
        weight_scale=WEIGHT_SCALE, block_k=variant.block_k, kv_block_size=KV_BLOCK_SIZE,
        num_warps=4 if variant.block_k == 256 else 1,
        out=fly, cta_info=cta, n_ctas=n_ctas,
    )  # fmt: skip
    torch.cuda.synchronize()
    col = torch.arange(msl, device=dev).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    want, got = fly[inside].float(), out[inside].float()
    if want.numel() == 0:
        return float("nan")
    atol = max(1e-2, REASSOC_ATOL_REL * float(want.abs().max()))
    return checkAllclose(
        want, got, rtol=2e-5, atol=atol, msg="vs flydsl", printLog=False
    )


# ── correctness ───────────────────────────────────────────────────────────────
def _variants():
    """Compiled instances for the current device. Looked up lazily, never at import: the probe
    inits HIP and raises on arches ``main()`` is meant to skip on other CI shards."""
    return pa_mqa_logits_mxfp4_variants()


def _qpb_max():
    """The widest compiled ``Q_PER_BLOCK`` (the row replication factor of ``_g4``)."""
    return max((v.q_per_block for v in _variants()), default=4)


def _resolve_variant(name):
    """``name`` if this arch compiled it, else ``None`` (the op default), so one decode shape
    list drives both targets."""
    if name is None:
        return None
    return name if name in {v.name for v in _variants()} else None


def assert_qshare_windows(cu_tiles, num_tiles, local_starts, local_ends, q_per_block):
    """Assert windows are non-decreasing within each tile, which the schedule assumes so a
    tile's union is first start .. last end. Host-side (syncs), so it lives in the test.
    ``q_per_block`` must be this plan's instance, or the span check is vacuous.
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
    # The buffers carry the instance, which is how a case pins one instead of the plan's pick.
    buffers = pa_mqa_logits_mxfp4_plan_buffers(
        dev, total_q, len(qlens), variant=variant
    )
    # Non-zero window starts need `local_starts`; `row_to_batch` because block_tables is per
    # sequence, not per query row.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, buffers=buffers, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    if check_windows and plan.variant.q_per_block > 1:
        # Unchecked by the kernel, and a violation deadlocks the CTA; moot at one row per tile.
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
    fly = flydsl_cross_check(inp, out, rb, ls, le, plan.variant)
    fly_ok = math.isnan(fly) or fly == 0
    return {
        "case": label, "variant": plan.variant.name,
        "rows": total_q, "tiles": plan.num_tiles, "ctas": plan.num_ctas,
        "max_win": int(le.max()), "err": err, "vs flydsl": fly, "oob -inf": oob,
        "window written": wr, "pass": err == 0 and oob and wr and fly_ok,
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
    """Replicate each row ``_qpb_max()`` times so a tile's rows share one window (the easy
    regime; the CSA rules below make adjacent rows differ)."""
    return [[r for r in b for _ in range(_qpb_max())] for b in windows_per_batch]


def _csa_fresh(qlen):
    """Fresh sequence: row n sees ``(n + 1) // RATIO``; runs start at n = 3 mod 4."""
    return [(0, (n + 1) // CSA_RATIO) for n in range(qlen)]


def _csa_chunked(qlen, kvlen):
    """``kvlen`` compressed rows committed and this chunk is the tail: row n sees
    ``kvlen - (qlen - 1 - n) // RATIO``, so runs align to the tail rather than to n."""
    return [(0, kvlen - (qlen - 1 - n) // CSA_RATIO) for n in range(qlen)]


# `cu_seq_q` claims 14 rows over a 10-row `q`: at 4 rows per tile the cut [8,12) straddles the
# bound (per-wave clause) and [12,14) is past it (CTA-uniform clause).
ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS, ROWID_WIN = 10, 14, 200
ROWID_CU = (0, 4, 8, ROWID_CLAIMED_ROWS)


def check_row_id_bound(data_init, scale_init, seed, variant):
    """Device-side ``cu_seq_q`` claims more rows than ``q`` holds (host checks all pass); the
    kernel must bound ``row_id`` by ``num_rows`` and drop the surplus. The oversized ``out``
    gives it teeth: surplus rows must stay -inf, where a right-sized one would absorb the
    overrun silently. Only multi-row tiles read ``cu_seq_q``; at one row it is a control.
    """
    real, claimed = ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS
    bs = len(ROWID_CU) - 1
    inp = build_inputs(bs, ROWID_WIN, real, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    # Windows and `out` cover `claimed` rows; `q` and `total_q` hold `real`.
    rb = t([b for b in range(bs) for _ in range(ROWID_CU[b + 1] - ROWID_CU[b])])
    ls = torch.zeros(claimed, dtype=torch.int32, device=dev)
    le = t([ROWID_WIN] * claimed)
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, claimed, bs, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        t(list(ROWID_CU)),
        le,
        buffers=buffers,
        total_q=real,
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

    rows = list(range(real))
    err = check_rows(out, ref_rows(inp, rows, rb, ls, le), "row_id bound")
    oob = oob_is_neginf(out[:real], ls[:real], le[:real])
    wr = window_is_written(out[:real], ls[:real], le[:real])
    # Fails without the kernel's bound: rows past `q` must stay untouched.
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


def check_raw_row_guard(data_init, scale_init, seed, variant):
    """Kernel row bound on both arches: a raw ``_launch`` of a 14-row schedule over a 10-row
    ``q`` with ``num_rows = 10`` must leave rows 10..13 of the oversized ``out`` at -inf.
    Unreachable through the public op, which passes the plan's own row count.
    """
    real, claimed = ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS
    inp = build_inputs(1, ROWID_WIN, real, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    rb = torch.zeros(claimed, dtype=torch.int32, device=dev)
    ls = torch.zeros(claimed, dtype=torch.int32, device=dev)
    le = t([ROWID_WIN] * claimed)
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, claimed, 1, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        t([0, claimed]), le, buffers=buffers, total_q=claimed, local_starts=ls, row_to_batch=rb
    )
    out = torch.full(
        (claimed, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )
    _launch(
        plan.variant, inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, plan.local_ends, plan.cta_info, plan.num_ctas, real, inp.max_seq_len,
        local_starts=plan.local_starts, weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE,
        out=out,
    )  # fmt: skip
    torch.cuda.synchronize()

    err = check_rows(out, ref_rows(inp, list(range(real)), rb, ls, le), "raw row guard")
    oob = oob_is_neginf(out[:real], ls[:real], le[:real])
    wr = window_is_written(out[:real], ls[:real], le[:real])
    untouched = bool(torch.isneginf(out[real:]).all().item())
    ret = {
        "data_init": data_init, "scale_init": scale_init, "seed": seed,
        "case": f"raw launch: cta_info {claimed} rows, num_rows {real}",
        "variant": plan.variant.name, "rows": real,
        "tiles": plan.num_tiles, "ctas": plan.num_ctas, "max_win": ROWID_WIN,
        "err": err, "oob -inf": oob and untouched, "window written": wr,
        "pass": err == 0 and oob and untouched and wr,
    }  # fmt: skip
    del inp, out
    torch.cuda.empty_cache()
    return ret


def check_row_count_raises(data_init, scale_init, seed, variant):
    """Host-visible row mismatches must raise before any kernel runs: a plan for 14 rows over a
    10-row ``q`` (checked in C++ ``fwd_sched``), and ``local_ends`` shorter than ``total_q``
    (checked in ``build_sched``).
    """
    real, claimed = ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS
    inp = build_inputs(1, ROWID_WIN, real, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    def raises(fn, needle):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- AITER_CHECK surfaces as RuntimeError
            return needle in str(e)
        return False

    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, claimed, 1, variant=variant)
    le = t([ROWID_WIN] * claimed)
    plan = pa_mqa_logits_mxfp4_plan(
        t([0, claimed]), le, buffers=buffers, total_q=claimed,
        row_to_batch=torch.zeros(claimed, dtype=torch.int32, device=dev),
    )  # fmt: skip
    q_short = raises(
        lambda: pa_mqa_logits_mxfp4(
            inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
            inp.weights, plan, inp.max_seq_len,
            weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE,
        ),
        "num_rows",
    )  # fmt: skip
    ends_short = raises(
        lambda: pa_mqa_logits_mxfp4_plan(
            t([0, claimed]), t([ROWID_WIN] * real), buffers=buffers, total_q=claimed
        ),
        "local_ends",
    )
    torch.cuda.synchronize()
    del inp
    torch.cuda.empty_cache()
    return {
        "data_init": data_init, "scale_init": scale_init, "seed": seed,
        "case": f"host raises: plan {claimed} > q {real}, local_ends {real} < total_q {claimed}",
        "variant": variant if isinstance(variant, str) else variant.name,
        "rows": claimed, "pass": q_short and ends_short,
    }  # fmt: skip


# ATOM's cudagraph decode metadata: batch padded to the captured size, `cu_seq_q` flat over
# the pad tail, pad rows with `row_to_batch = -1` and empty windows. Contexts cross tile edges.
ATOM_REAL_CTX = (300, 70, 129, 1, 257)
ATOM_PAD_SEQS = 8


def check_atom_decode(data_init, scale_init, seed, variant, mtp, pad_seqs):
    """One ATOM-shaped decode: live sequences of ``mtp`` rows padded to ``pad_seqs``. Pad rows
    must stay -inf (pre-filled only to make that visible). ``pad_seqs == live`` leaves surplus
    tiles at ``r0 == total_q``, one past the end of ``row_to_batch``.
    """
    live = len(ATOM_REAL_CTX)
    total_q = pad_seqs * mtp
    real_q = live * mtp
    inp = build_inputs(live, max(ATOM_REAL_CTX), total_q, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    # MTP tail-causal: row j of a sequence sees `ctx - (mtp - 1 - j)`.
    rb = t([b for b in range(live) for _ in range(mtp)] + [-1] * (total_q - real_q))
    ls = torch.zeros(total_q, dtype=torch.int32, device=dev)
    le = t(
        [max(c - (mtp - 1 - j), 0) for c in ATOM_REAL_CTX for j in range(mtp)]
        + [0] * (total_q - real_q)
    )
    cu = t([b * mtp for b in range(live + 1)] + [real_q] * (pad_seqs - live))
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, pad_seqs, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, buffers=buffers, total_q=total_q, row_to_batch=rb
    )
    out = torch.full(
        (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )
    pa_mqa_logits_mxfp4(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, plan, inp.max_seq_len,
        weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE, out=out,
    )  # fmt: skip
    torch.cuda.synchronize()

    rows = [r for r in range(real_q) if int(le[r]) > 0]
    err = check_rows(out, ref_rows(inp, rows, rb, ls, le), "atom decode")
    # Over EVERY row, pad rows included: an empty window leaves the whole row at -inf.
    oob = oob_is_neginf(out, ls, le)
    wr = window_is_written(out, ls, le)
    ret = {
        "data_init": data_init, "scale_init": scale_init, "seed": seed,
        "case": f"atom decode mtp={mtp} {live}/{pad_seqs} seqs", "variant": plan.variant.name,
        "rows": total_q, "tiles": plan.num_tiles, "ctas": plan.num_ctas,
        "max_win": int(le.max()), "err": err, "oob -inf": oob, "window written": wr,
        "pass": err == 0 and oob and wr,
    }  # fmt: skip
    del inp, out
    torch.cuda.empty_cache()
    return ret


def run_corner(data_init, scale_init, seed):
    """Corner cases (short groups, offset starts, tile edges, starts mod 128, CSA regimes) on
    every compiled instance, plus the row-count probes and ATOM decode. Case seeds are offsets
    from ``--seed``.
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
    # Every compiled instance, pinned: left to the plan, these shapes would never pick one-row.
    variants = _variants()
    if not variants:
        raise RuntimeError(
            "no compiled kernel instances to run the corner suite on; an empty sweep reports "
            "`pass` having tested nothing"
        )
    # The cu_seq_q probe is gfx1250-only (only its tile cut reads cu_seq_q); the others run on both.
    row_id = (
        []
        if _is_permuted()
        else [check_row_id_bound(data_init, scale_init, seed + 70, v) for v in variants]
    ) + [
        check(data_init, scale_init, seed + 72, v)
        for check in (check_row_count_raises, check_raw_row_guard)
        for v in variants
    ]
    atom = [
        check_atom_decode(data_init, scale_init, seed + 90 + mtp, v, mtp, pad)
        for v in variants
        for mtp, pad in ((1, ATOM_PAD_SEQS), (2, ATOM_PAD_SEQS), (2, len(ATOM_REAL_CTX)))
    ]
    return [
        check_prefill(w, seed + case_seed, label, data_init, scale_init, v)
        for v in variants
        for w, case_seed, label in cases
    ] + row_id + atom


SPREAD_PROBE_ROWS = 4096


def scale_spread(scale_init, seed=0, rows=SPREAD_PROBE_ROWS):
    """``(distinct exponents, fraction of rows 16 apart that disagree)`` for a ``--scale-init``.
    The second decides: a lane-half misroute is invisible to a spread periodic in 16."""
    e8 = fill_exponents(
        (rows, BLOCKS_ROW), scale_init, make_generator(seed, device=dev)
    )
    block0 = e8[:, 0]
    distinct = int(torch.unique(e8).numel())
    disagree = float((block0[:-16] != block0[16:]).float().mean().item())
    return distinct, disagree


def correctness_blindness(data_init, scale_init):
    """Why this init pair cannot catch a wrong answer (all-zero data, or too little exponent
    spread), or ``None`` when it can. Such pairs are skipped with the reason, not passed."""
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
    """A copy of ``kv_scale`` with KV row ``row_in_seq`` of every batch set to 0xFF (E8M0 NaN).
    The mark is set in the natural layout and pushed through the arch's layout transform rather
    than by computing byte offsets by hand.
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
    """A NaN E8M0 scale must reach exactly the in-window cells at its KV row: the relu must
    propagate NaN, not swallow it. Asserted as an exact set, so over-smearing also fails.
    """
    total_q = bs * next_n
    # Raised, not asserted, so `python -O` keeps it.
    if len(ends) != total_q:
        raise ValueError(f"{label}: {len(ends)} window ends for {total_q} rows")
    inp = build_inputs(bs, max(max(ends), 1), total_q, seed, "norm", "auto")

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    rb = t([b for b in range(bs) for _ in range(next_n)])
    ls = torch.zeros(total_q, dtype=torch.int32, device=dev)
    le = t(list(ends))
    cu = t([0] + list(itertools.accumulate([next_n] * bs)))
    kvs = poison_kv_rows(inp.kv_scale, inp.block_tables, kv_row)

    # Prefill passes (zero) `local_starts`, decode passes None: same schedule, different branch.
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
    """:func:`check_nan_scale` over both entry points and every compiled instance. Separate from
    :func:`run_corner` because these cases require non-finite in-window cells."""
    oks = []
    for v in _variants():
        # Ragged windows, all containing KV row 0.
        oks.append(check_nan_scale("prefill", 2, 3, [50, 120, 200, 40, 100, 180],
                                   seed + 80, v, "prefill, nan at kv row 0"))  # fmt: skip
        # KV row 2: only row 3's window [0,3) holds it; rows 1-2 end at or before it.
        oks.append(check_nan_scale("prefill", 1, 4, [0, 1, 2, 3],
                                   seed + 81, v, "prefill, nan at kv row 2", kv_row=2))  # fmt: skip
        # Control: no window holds the poisoned row, so any non-finite cell fails.
        oks.append(check_nan_scale("prefill", 1, 4, [1, 2, 2, 2],
                                   seed + 83, v, "prefill, nan outside every window",
                                   kv_row=2))  # fmt: skip
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
    """MTP tail-causal windows in packed (b, n) order: row n of batch b sees
    ``[0, ctx[b] - (qlen[b] - 1 - n))``; plain causal when ``qlen == ctx``."""
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
    """Time the launch, then score it. Scoring runs after timing and frees its temporaries:
    the reference's ~1 GB of allocator churn would otherwise skew the next timing."""
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
    """Causal prefill: 16384 rows split across ``bs`` batches, ctx == qlen. The split is seeded
    by ``bs`` so ``--seed`` moves only the data."""
    qlens = gen_prefill_qlens(bs, seed=bs)
    total_q = sum(qlens)
    inp = build_inputs(bs, max(qlens), total_q, seed, data_init, scale_init)
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    rb, ls, le = tail_causal_windows(qlens, qlens)
    # The plan is per forward, so it is built outside the timed region.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    out = torch.full(
        (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    # Bound as defaults, not closed over, to avoid late binding across shapes.
    def ours(inp=inp, plan=plan, out=out):
        return pa_mqa_logits_mxfp4(
            inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
            inp.weights, plan, inp.max_seq_len,
            weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE, out=out,
        )  # fmt: skip

    n_logits = int((le - ls).clamp(min=0).sum().item())
    ret = {
        "gfx": get_gfx(),
        # The op's own default instance (no `variant` passed).
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


def run_windowed_case(per_batch, data_init, scale_init, seed, variant=None):
    """Launch and score one prefill case from explicit per-row ``(start, end)`` windows, on the
    op's default instance or on ``variant``. Shared by the two CSA regimes."""
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
    # Built outside the timed region, as in `test_prefill_causal`.
    buffers = (
        None
        if variant is None
        else pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, bs, variant=variant)
    )
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, buffers=buffers, total_q=total_q, local_starts=ls, row_to_batch=rb
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
    """Fresh CSA prefill: ``bs`` empty sequences of ``qlen`` rows, row n sees
    ``(n + 1) // CSA_RATIO``."""
    return run_windowed_case(
        [_csa_fresh(qlen) for _ in range(bs)], data_init, scale_init, seed
    )


@benchmark()
def test_prefill_chunked(bs, kvlen, variant, data_init, scale_init, seed):
    """Chunked CSA prefill at PR #5332's shapes: ``PREFILL_TOTAL_QLEN`` rows split raggedly by
    ``gen_prefill_qlens``, each chunk the tail of ``kvlen`` compressed rows, so row n sees
    ``kvlen - (qlen - 1 - n) // CSA_RATIO``. ``variant`` None means the op default.
    """
    return run_windowed_case(
        [_csa_chunked(q, kvlen) for q in gen_prefill_qlens(bs, seed=bs)],
        data_init,
        scale_init,
        seed,
        variant,
    )


@benchmark()
def test_decode(mtp, seqs, n_long, variant, data_init, scale_init, seed):
    """MTP decode: ``seqs`` sequences of ``mtp`` rows, ``n_long`` with the long window, on
    ``variant`` (None = op default). Tile count is fixed per ``(mtp, seqs)``, so the ragged
    shapes isolate load balance. Every row takes its sequence's whole window.
    """
    variant = _resolve_variant(variant)  # the arch default if not compiled here
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
    # `local_starts` stays None, as in ATOM's decode call.
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
    """One result table as markdown (to read) and one-line JSON (for a benchmark driver)."""
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", name, df.to_markdown(index=False))
    print_json_table(name, df)


def init_pairs(data_init, scale_init):
    """Pair the two axes position-wise (not crossed, matching the benchmark driver); a length-1
    side broadcasts."""
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
    # Arch gate before anything touches the device: CI runs this file on every shard.
    if get_gfx() not in SUPPORTED_GFX:
        why = f"built for {'/'.join(SUPPORTED_GFX)}, skipped on {get_gfx()}"
        aiter.logger.warning("MXFP4 MQA logits: %s", why)
        # Still emit a table, so a driver sees a skip rather than a broken extractor.
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
    # Not `add_data_init_args`: its --scale-init offers float dists, not E8M0_SCALE_DISTS.
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
            # Own table: a skip row would NaN-fill and turn the bool columns into floats.
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

    # NaN E8M0 propagation, on its own data, independent of the init pairs.
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

    # PR #5332's chunked shapes, on the default and every other compiled instance (gfx950's
    # `qlen1_kv256` is aimed at these long windows).
    chunked_shapes = [(1, 25000), (2, 25000), (4, 25000)]
    default = _default_variant(_device_arch(torch.cuda.current_device()))
    chunked_variants = [None] + [v.name for v in _variants() if v.name != default]
    summarize(
        f"pa_mqa_logits_mxfp4 prefill chunked (csa ratio {CSA_RATIO}, #5332 shapes)",
        [
            test_prefill_chunked(bs, kvlen, v, data_init, scale_init, args.seed)
            for data_init, scale_init in pairs
            for v in chunked_variants
            for bs, kvlen in chunked_shapes
        ],
    )

    # (mtp, seqs, n_long, variant). MTP = 1 carries the batch sweep: there `seqs` is the tile
    # count. Ragged rows hold the tile count and move only the long fraction. mtp = 1 names the
    # one-row instance explicitly; the op default (four-row on gfx1250) would idle 3 of 4 waves.
    decode_shapes = [
        (1, 1, 1, "qlen1_kv64"),
        (1, 8, 8, "qlen1_kv64"),
        (1, DECODE_SEQS, DECODE_SEQS, "qlen1_kv64"),
        (1, 128, 128, "qlen1_kv64"),
        (1, DECODE_SEQS, 4, "qlen1_kv64"),
        (1, 128, 16, "qlen1_kv64"),
        # gfx950-only instance; gfx1250 falls back to its default (`_resolve_variant`).
        (1, DECODE_SEQS, DECODE_SEQS, "qlen1_kv256"),
        (1, 128, 16, "qlen1_kv256"),
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
