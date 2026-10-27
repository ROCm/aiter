# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS) through ``aiter.ops.opus.pa_mqa_logits_mxfp4``: perf and
correctness.

    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py                    # every table
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py --modes decode -m v41 -b 1 64 --mtp 6
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py --modes semantics --checks edges graph

Models (``-m``): ``v4`` (DeepSeek-V4, 64 index heads, compression ratio 4, 64-row pages, one a
table entry), ``v41`` (V4.1 full layers, 32 heads, 8-row pages, a table entry a 256-token PAGE
of them, ratio ``--ratio``) and ``cand`` (V4.1 candidate layers: every row its own sequence and
its own table of 8-row pages).

``decode`` / ``prefill``: every registered instance that takes the model, in every schedule mode it
compiles (``table``: a schedule built once outside the timing, as V4 builds one a forward;
``internal``: no schedule, built in the call; ``free`` / ``prefill``: schedule-free, the in-kernel
assignment timed), against an f64-GEMM reference on sampled rows; ``default`` names the
config's own choice. Prefill shapes are 16384 rows (ATOM's batched-token budget).

``semantics`` (``--checks``), each against the reference on every instance and mode:
  rows     every row of small shapes, NaN where an E8M0 is 0xFF, nothing written outside a
           window, past ``out``'s guard bands or to an input; the flat out (``out_offsets``)
           equal to the plane bit for bit
  edges    rows carrying one source of NaN / Inf each (0xFF scales, a NaN or Inf weight, products
           that overflow, a NaN key behind a zero q block): NaN / +-Inf masks equal the
           reference's IEEE ones. The kernels compute relu as max_num and put NaN back from the
           scales.
  bands    a forward scored in slices of rows (``query_start_loc`` clamped) equals it whole
  padding  rows past ``query_start_loc[-1]`` and empty sequences untouched, the rest right
  graph    schedule + launch captured in a CUDA graph and replayed on new data
  errors   the interface refuses what it cannot run (layouts, bounds, a foreign schedule, ...)
"""

import argparse
import bisect
import itertools
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus import pa_mqa_logits_mxfp4 as M
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
    mqa_logits_mxfp4,
    mqa_logits_mxfp4_config,
    mqa_logits_mxfp4_schedule,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950", "gfx1250"]

D = 128
PAGE_ROWS = 8  # the index plane's page
PAGE_TOKENS = 256  # ATOM's KV PAGE: a full layer's table entry spans it, compressed
CANDIDATE_BLOCKS = 2048
TILE = 64
PREFILL_ROWS = 16384
SAMPLE_ROWS = 16  # rows checked against the reference in a perf shape
REL_TOL = 1e-5  # of the largest finite |logit|; a misrouted E8M0 moves cells by 2x
MODEL_HEADS = {"v4": 64, "v41": 32, "cand": 32}

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1 = _E2M1_MAGNITUDES + tuple(
    -x for x in _E2M1_MAGNITUDES
)  # code -> value; bit 3 the sign


# == workloads: ATOM's rows, tables and natural FP4 data ===========================================
@dataclass
class Rows:
    """One scorer call: sequences of rows (``query_start_loc``), prefix windows and a table."""

    label: str
    kind: str  # "full" | "candidate"
    query_start_loc: torch.Tensor  # [seqs + 1] i32
    row_ends: torch.Tensor  # [rows] i32
    # full: [seqs, entries] of ppb pages; candidate: [rows, 2048] pages
    block_tables: torch.Tensor
    pages_per_block: int
    width: int  # logits columns
    num_pages: int  # 8-row pages in the pool

    @property
    def rows(self):
        return self.row_ends.numel()

    @property
    def num_seqs(self):
        return self.query_start_loc.numel() - 1

    @property
    def max_qlen(self):
        return max(1, int(self.query_start_loc.diff().max()))

    def key_rows(self, r):
        """Plane rows of row r's keys [0, row_end), in column order."""
        end = int(self.row_ends[r])
        k = torch.arange(end)
        seq = bisect.bisect_right(self.query_start_loc.tolist(), r) - 1
        per = self.pages_per_block * PAGE_ROWS
        return self.block_tables[seq, k // per].long() * per + k % per


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _tables(ctx_keys, rows_per_entry, g):
    need = [-(-n // rows_per_entry) for n in ctx_keys]
    ids = torch.randperm(sum(need), generator=g, device="cuda").to(torch.int32)
    table = torch.zeros(len(need), max(need), dtype=torch.int32)
    at = 0
    for b, n in enumerate(need):
        table[b, :n] = ids[at : at + n]
        at += n
    return table, sum(need)


def full_layer(label, ctx, rows_per_seq, ratio, *, seed=0):
    """Sequences whose last token sits at ``ctx[b] - 1``, each scoring its last
    ``rows_per_seq[b]`` tokens at compression ``ratio`` (row at position p sees (p + 1) // ratio
    keys). A table entry is one 256-token PAGE: 256 / ratio keys."""
    g = _gen(seed)
    keys = [c // ratio for c in ctx]
    ppb = PAGE_TOKENS // ratio // PAGE_ROWS
    table, entries = _tables(keys, ppb * PAGE_ROWS, g)
    ends, qsl = [], [0]
    for c, n in zip(ctx, rows_per_seq):
        ends += [(p + 1) // ratio for p in range(c - n, c)]
        qsl.append(qsl[-1] + n)
    return Rows(
        label,
        "full",
        torch.tensor(qsl, dtype=torch.int32),
        torch.tensor(ends, dtype=torch.int32),
        table,
        ppb,
        table.shape[1] * ppb * PAGE_ROWS,
        entries * ppb,
    )


def candidate_layer(label, ctx, rows_per_seq, *, seed=0):
    """V4.1's candidate layers: every row its own table of 8-row pages (picked from its request's
    pages, the newest last) and its own sequence."""
    g = _gen(seed)
    table, blocks = _tables(ctx, PAGE_TOKENS, g)
    ppb = PAGE_TOKENS // PAGE_ROWS
    tabs, ends = [], []
    for b, (c, n) in enumerate(zip(ctx, rows_per_seq)):
        nblk = -(-c // PAGE_ROWS)
        j = torch.arange(nblk)
        phys = table[b, j // ppb].long() * ppb + j % ppb
        base = torch.randperm(max(nblk - 1, 0), generator=g, device="cuda")[
            : CANDIDATE_BLOCKS - 1
        ]
        base = torch.sort(base).values
        for p in range(c - n, c):
            newest = -(-(p + 1) // PAGE_ROWS) - 1
            pick = base[: int(torch.searchsorted(base, newest))]
            kept = torch.cat([pick, torch.tensor([newest])])
            row = torch.zeros(CANDIDATE_BLOCKS, dtype=torch.int32)
            row[: kept.numel()] = phys[kept].to(torch.int32)
            tabs.append(row)
            ends.append((kept.numel() - 1) * PAGE_ROWS + (p + 1) - newest * PAGE_ROWS)
    rows = len(ends)
    return Rows(
        label,
        "candidate",
        torch.arange(rows + 1, dtype=torch.int32),
        torch.tensor(ends, dtype=torch.int32),
        torch.stack(tabs),
        1,
        CANDIDATE_BLOCKS * PAGE_ROWS,
        blocks * ppb,
    )


@dataclass
class Fp4Data:
    # [num_pages * 8, 64] u8: a key's 64 bytes, even element in the low nibble
    values: torch.Tensor
    # [num_pages * 8, 4] u8: its four E8M0 bytes, block c = dims [32c, 32c+32)
    scales: torch.Tensor
    q: torch.Tensor  # [rows, H, 64] u8
    q_scale: torch.Tensor  # [rows, H, 4] u8
    weights: torch.Tensor  # [rows, H] bf16
    weight_scale: float


def make_data(rows: Rows, heads, *, seed=1):
    """Random e2m1 codes and E8M0 exponents near 1 (no product overflows fp32); exponents vary
    row to row, so a scale applied to the wrong 32 elements or lane half shows."""
    g = _gen(seed)
    u8 = {"dtype": torch.uint8, "generator": g}
    keys, n = rows.num_pages * PAGE_ROWS, rows.rows
    return Fp4Data(
        values=torch.randint(0, 256, (keys, D // 2), **u8),
        scales=torch.randint(120, 128, (keys, D // 32), **u8),
        q=torch.randint(0, 256, (n, heads, D // 2), **u8),
        q_scale=torch.randint(124, 130, (n, heads, D // 32), **u8),
        weights=torch.randn(n, heads, generator=g).to(torch.bfloat16),
        weight_scale=(D**-0.5) * (heads**-0.5),
    )


def dequant(values, scales):
    """[..., D/2] e2m1 + [..., D/32] E8M0 -> [..., D] f64, 0xFF a NaN block. f64 so an E8M0 of
    2^127 times a 6 does not overflow before the dot product does."""
    lut = torch.tensor(E2M1, dtype=torch.float64)
    codes = torch.stack([values & 0xF, values >> 4], dim=-1).flatten(-2).long()
    scale = torch.where(scales == 0xFF, float("nan"), torch.exp2(scales.double() - 127))
    return lut[codes] * scale.repeat_interleave(32, dim=-1)


def run_torch(rows: Rows, data, r):
    """Row r's logits over [0, row_end): sum_h relu(q_h . k) * w_h * weight_scale. The dot product
    in f64, rounded to f32 as the WMMA's is; relu and the sum IEEE, so NaN / Inf land where the
    kernel's do. gfx950's MFMA rounds after each 32-element block instead: it adds the block's
    exact product to its fp32 accumulator, so an Inf from one block sticks where the f64 sum would
    not overflow.
    """
    kr = rows.key_rows(r)
    k = dequant(data.values[kr], data.scales[kr])
    q = dequant(data.q[r], data.q_scale[r])
    if _permuted():
        blk = torch.einsum(
            "hbk,ebk->heb", q.unflatten(-1, (4, 32)), k.unflatten(-1, (4, 32))
        )
        dot = blk[..., 0].float()
        for b in range(1, 4):
            dot = (dot.double() + blk[..., b]).float()
    else:
        dot = (q @ k.T).float()
    s = torch.relu(dot)  # [H, end]
    w = data.weights[r].float() * data.weight_scale
    return (w[:, None] * s).sum(0)


def inject_nans(rows: Rows, data):
    """E8M0 0xFF where V4.1 has it -- the first compressed row of every sequence, all four bytes --
    plus one byte of a key in the middle of every window and one q scale byte of the last row.
    """
    for r in range(rows.rows):
        kr = rows.key_rows(r)
        if kr.numel():
            data.scales[kr[0]] = 0xFF
            data.scales[kr[kr.numel() // 2], 2] = 0xFF
    data.q_scale[rows.rows - 1, 5, 1] = 0xFF


# == a call as the model makes it, every way the interface can run it ===============================
def _permuted():
    """gfx950 reads MFMA-permuted q_scale / kv_cache / kv_scale; gfx1250 the natural ones."""
    return get_gfx() == "gfx950"


def _scale_to_gfx950(e8, rows_per_group):
    """[rows, 4] natural E8M0 -> gfx950's [rows / 64, 2 (g), 32 (m), 4 (kt, tile)], a lane's four
    bytes in one dword: byte kt * 2 + tile of dword (g, m) is row tile * 32 + m's block
    kt * 2 + g. ``rows_per_group``: 64 heads (q_scale) or a 64-row page (kv_scale)."""
    groups = e8.shape[0] // rows_per_group
    return (
        e8.reshape(groups, rows_per_group // 32, 32, 2, 2)
        .permute(0, 4, 2, 3, 1)
        .reshape(groups, 2, 32, 4)
        .contiguous()
    )


@dataclass
class Case:
    rows: Rows
    heads: int
    data: Fp4Data
    page_size: int  # the plane as kv_cache [pages, page_size, 64]
    pages_per_block: int
    block_tables: torch.Tensor
    # gfx950's permuted (q_scale, kv, kv_scale), made on first use
    _arch: tuple | None = None

    def _layout(self):
        d = self.data
        if not _permuted():
            return (
                d.q_scale,
                d.values.view(-1, self.page_size, D // 2),
                d.scales.view(-1, self.page_size, D // 32),
            )
        # kv_cache[page, b, o, :] holds key o's 32-element block b, 16 packed bytes
        kv = (
            d.values.view(-1, self.page_size, D // 32, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return (
            _scale_to_gfx950(d.q_scale.reshape(-1, D // 32), self.heads),
            kv,
            _scale_to_gfx950(d.scales, self.page_size),
        )

    def arch_inputs(self):
        """(q_scale, kv_cache, kv_scale) in the device's layout. gfx950's are permuted copies of
        the data as it was at the first call: after editing the data in place, refresh().
        """
        if self._arch is None or not _permuted():
            self._arch = self._layout()
        return self._arch

    def refresh(self):
        """Copy edited data into the permuted tensors already handed out (a captured graph
        reads those)."""
        if self._arch is not None:
            for dst, src in zip(self._arch, self._layout()):
                dst.copy_(src)

    @property
    def q_scale(self):
        return self.arch_inputs()[0]

    @property
    def kv(self):
        return self.arch_inputs()[1]

    @property
    def kv_scale(self):
        return self.arch_inputs()[2]


def make_case(rows: Rows, heads, *, page64=False, nan=False, seed=1):
    """``page64``: the table as 64-row pages, one an entry (ATOM's V4 layout)."""
    data = make_data(rows, heads, seed=seed)
    if nan:
        inject_nans(rows, data)
    if not page64:
        return Case(
            rows, heads, data, PAGE_ROWS, rows.pages_per_block, rows.block_tables
        )
    tpb = rows.pages_per_block * PAGE_ROWS // TILE
    s = tpb.bit_length() - 1
    t = torch.arange(rows.block_tables.shape[1] * tpb)
    table = ((rows.block_tables[:, t >> s].long() << s) + (t & (tpb - 1))).to(
        torch.int32
    )
    return Case(rows, heads, data, TILE, 1, table.contiguous())


def config(case: Case, **kw):
    r = case.rows
    return mqa_logits_mxfp4_config(
        torch.cuda.current_device(),
        heads=case.heads,
        page_size=case.page_size,
        pages_per_block=case.pages_per_block,
        max_rows=r.rows,
        max_seqs=r.num_seqs,
        max_qlen=r.max_qlen,
        max_seq_len=r.width,
        **kw,
    )


def launch(
    case: Case,
    cfg,
    out,
    *,
    schedule=None,
    out_offsets=None,
    qsl=None,
    ends=None,
    rows=None,
):
    """One call over the case's rows, or over `rows` (a slice) with its own bounds."""
    d = case.data
    sl = slice(None) if rows is None else rows
    return mqa_logits_mxfp4(
        d.q[sl],
        case.q_scale[sl],
        case.kv,
        case.kv_scale,
        d.weights[sl],
        case.block_tables,
        case.rows.query_start_loc if qsl is None else qsl,
        case.rows.row_ends if ends is None else ends,
        cfg=cfg,
        schedule=schedule,
        weight_scale=d.weight_scale,
        out=out,
        out_offsets=out_offsets,
    )


def launch_scheduled(case: Case, cfg, out, schedule):
    return launch(case, cfg, out, schedule=schedule)


def ways(case: Case):
    """{name: (cfg, build a schedule?)}: every registered instance that takes the case, in every
    mode it compiles -- Table from a schedule, Table from the internal schedule (an instance
    without a schedule-free mode), Free, Prefill."""
    entry_rows = case.page_size * case.pages_per_block
    out = {}
    if not takes(case):
        return out
    for inst in M._instances(torch.device("cuda", torch.cuda.current_device())):
        if inst.heads != case.heads or not inst.reads(entry_rows):
            continue
        cfg = config(case, _instance=inst.id, _mode="table")
        out[f"{inst.name}:table"] = (cfg, True)
        modes = [m for m in ("free", "prefill") if inst.has(M._MODE_NAMES[m])]
        if not modes:
            out[f"{inst.name}:internal"] = (cfg, False)
        for m in modes:
            out[f"{inst.name}:{m}"] = (config(case, _instance=inst.id, _mode=m), False)
    return out


def takes(case: Case):
    """Whether this arch's KV layout exists at the case's page size (gfx950: 64 rows only)."""
    return M._PAGE_SIZE.get(get_gfx(), case.page_size) == case.page_size


def serves(model):
    """Whether this op serves `model` on this arch (gfx950 sends V4.1 to FlyDSL)."""
    return (get_gfx(), MODEL_HEADS[model]) not in M._ELSEWHERE


def default_way(case: Case):
    """The config's own choice for the case, or "none"."""
    try:
        cfg = config(case)
    except ValueError:
        return "none"
    mode = {None: "internal", M.FREE: "free", M.PREFILL: "prefill"}[cfg._mode]
    return f"{cfg.instance_name}:{mode}"


def schedule_for(cfg, case, build, qsl=None, ends=None):
    if not build:
        return None
    r = case.rows
    return mqa_logits_mxfp4_schedule(
        cfg,
        r.query_start_loc if qsl is None else qsl,
        r.row_ends if ends is None else ends,
        buffers=cfg.schedule_buffers(),
    )


def row_errors(got, ref):
    """Why row `got` [end] is not `ref`: the NaN / +-Inf masks first, then the finite cells."""
    for name, f in (
        ("NaN", torch.isnan),
        ("+Inf", torch.isposinf),
        ("-Inf", torch.isneginf),
    ):
        if not torch.equal(f(got), f(ref)):
            return f"{name} at {int(f(got).sum())} cells, want {int(f(ref).sum())}"
    fin = torch.isfinite(ref)
    if not bool(fin.any()):
        return None
    scale = ref[fin].abs().max().clamp_min(1e-30)
    err = ((got[fin] - ref[fin]).abs().max() / scale).item()
    return f"max rel err {err:.1e}" if err > REL_TOL else None


# == decode / prefill: perf of every way, checked on sampled rows ===================================
def model_rows(model, ratio, label, ctx, rows_per_seq):
    if model == "cand":
        return candidate_layer(label, ctx, rows_per_seq)
    return full_layer(label, ctx, rows_per_seq, 4 if model == "v4" else ratio)


def run_ways(model, rows: Rows):
    """Time and check every way of `model` on `rows`; the @benchmark tests' common body."""
    heads = MODEL_HEADS[model]
    case = make_case(rows, heads, page64=model == "v4")
    case.arch_inputs()  # gfx950's permuted copies, made outside the timing
    n = rows.rows
    sample = sorted(
        {int(x) for x in torch.linspace(0, n - 1, min(n, SAMPLE_ROWS)).round()}
    )
    refs = [run_torch(rows, case.data, r) for r in sample]
    ref_cat = torch.cat(refs)
    tol = REL_TOL * float(ref_cat.abs().max())
    row_keys = int(rows.row_ends.long().sum())
    # Keys read at least once (a sequence's rows share its keys; a candidate row shares nothing),
    # each row's q / scales / weights, and the logits written.
    ends, qsl = rows.row_ends.long(), rows.query_start_loc.tolist()
    keys = (
        row_keys
        if rows.kind == "candidate"
        else sum(int(ends[a:b].max()) for a, b in itertools.pairwise(qsl) if b > a)
    )
    flops = 2 * heads * D * row_keys
    nbytes = (
        keys * (D // 2 + D // 32) + n * heads * (D // 2 + D // 32 + 2) + row_keys * 4
    )

    ret = {"gfx": get_gfx(), "rows": n, "default": default_way(case)}
    # Preallocated, as the model passes it.
    out = torch.empty(n, rows.width, dtype=torch.float32)
    for name, (cfg, build) in ways(case).items():
        # V4 builds the schedule once a forward, outside the layers.
        schedule = schedule_for(cfg, case, build)
        out.fill_(float("-inf"))
        # Positional, so run_perftest's per-rotation deepcopy of the arguments keeps the schedule
        # bound to the very query_start_loc / row_ends tensors of its copy of the case.
        _, us = run_perftest(
            launch_scheduled, case, cfg, out, schedule, num_iters=50, num_warmup=10
        )
        got = torch.cat([out[r, : ref.numel()] for r, ref in zip(sample, refs)])
        err = checkAllclose(
            ref_cat, got, rtol=REL_TOL, atol=tol, msg=f"{rows.label} {name}: "
        )
        tails = [out[r, ref.numel() :] for r, ref in zip(sample, refs)]
        if not all(bool(torch.isneginf(t).all()) for t in tails):
            aiter.logger.error("%s %s: wrote past a row's window", rows.label, name)
            err = 1.0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_decode(model, ratio, batch, ctx, mtp):
    """`batch` sequences at context `ctx`, each scoring its last `mtp` tokens."""
    rows = model_rows(model, ratio, f"{model} dec", [ctx] * batch, [mtp] * batch)
    return run_ways(model, rows)


@benchmark()
def test_prefill(model, ratio, qlen, ctx):
    """16384 rows: 16384 // `qlen` sequences, each a `qlen`-token chunk ending at `ctx` (fresh
    when ctx == qlen)."""
    batch = PREFILL_ROWS // qlen
    rows = model_rows(
        model, ratio, f"{model} pre", [max(ctx, qlen)] * batch, [qlen] * batch
    )
    return run_ways(model, rows)


# == semantics ======================================================================================
def rows_shapes(model):
    """(rows, page64, nan) small enough to check every row."""
    if model == "v4":
        shapes = [
            (full_layer("v4 dec mtp1", [2048, 1500, 777], [1, 1, 1], 4), False),
            (full_layer("v4 dec mtp4 ragged", [9000, 1500, 777], [4, 3, 1], 4), False),
            (full_layer("v4 dec r2 6 rows", [3000, 2048], [6, 6], 2), False),
            (
                full_layer("v4 dec tiers", [2048, 1500, 777, 3000], [2, 4, 6, 8], 4),
                False,
            ),
            (full_layer("v4 dec r1 short", [70], [6], 1), False),
            (full_layer("v4 pre fresh 600", [600], [600], 4), False),
            (full_layer("v4 pre chunk 300@5000", [5300], [300], 4), False),
            (full_layer("v4 mixed", [5000, 100, 2600], [6, 37, 1], 4), False),
            (full_layer("v4 nan dec", [2048, 1500, 777], [4, 4, 1], 4), True),
            (full_layer("v4 nan pre r2 200@1000", [1200], [200], 2), True),
        ]
        out = []
        for rows, nan in shapes:
            out.append((rows, True, nan))
            # ratio 4: also 8-row pages, 8 an entry
            if rows.pages_per_block * PAGE_ROWS == TILE:
                out.append((rows, False, nan))
        return out
    if model == "v41":
        return [
            (rows, False, nan)
            for rows, nan in [
                (
                    full_layer("v41 dec r2 ragged", [2048, 1500, 777], [6, 4, 1], 2),
                    False,
                ),
                (
                    full_layer(
                        "v41 dec r1 mtp1", [3000, 2048, 777, 70], [1, 1, 1, 1], 1
                    ),
                    False,
                ),
                (full_layer("v41 dec r1", [3000, 2048], [6, 6], 1), False),
                (
                    full_layer(
                        "v41 dec r2 tiers", [2048, 1500, 777, 3000], [2, 3, 4, 8], 2
                    ),
                    False,
                ),
                (full_layer("v41 dec r1 short", [70], [6], 1), False),
                (full_layer("v41 pre r2 fresh 300", [300], [300], 2), False),
                (full_layer("v41 pre r1 chunk 200@1000", [1200], [200], 1), False),
                (full_layer("v41 mixed r1", [5000, 100, 2600], [6, 37, 1], 1), False),
                (full_layer("v41 nan dec r2", [2048, 1500, 777], [6, 4, 1], 2), True),
                (full_layer("v41 nan pre r1 200@1000", [1200], [200], 1), True),
            ]
        ]
    return [
        (rows, False, nan)
        for rows, nan in [
            (candidate_layer("cand bs2 ctx20K", [20000, 17000], [6, 6]), False),
            (candidate_layer("cand short 3K", [3000], [6]), False),
            (
                candidate_layer("cand ragged 40K", [40000, 18000, 70000], [6, 2, 1]),
                False,
            ),
            (candidate_layer("cand nan bs2", [20000, 17000], [6, 6]), True),
        ]
    ]


def check_rows(model):
    """Every row of small shapes on every way; returns [(case label, failures)]."""
    results = []
    for rows, page64, nan in rows_shapes(model):
        case = make_case(rows, MODEL_HEADS[model], page64=page64, nan=nan)
        if not takes(case):
            continue
        inputs = (
            case.data.values,
            case.data.scales,
            case.data.q,
            case.data.q_scale,
            case.data.weights,
            case.block_tables,
            rows.row_ends,
            rows.query_start_loc,
        )
        before = [x.clone() for x in inputs]
        refs = [run_torch(rows, case.data, r) for r in range(rows.rows)]
        tag = f"{rows.label} page{case.page_size}x{case.pages_per_block}"
        # An instance's Table logits: its other modes must equal them bit for bit.
        table_out = {}
        for name, (cfg, build) in ways(case).items():
            guard = rows.width * 4
            store = torch.full((rows.rows * rows.width + 2 * guard,), float("-inf"))
            out = store[guard : guard + rows.rows * rows.width].view(
                rows.rows, rows.width
            )
            schedule = schedule_for(cfg, case, build)
            launch(case, cfg, out, schedule=schedule)
            flat = None
            if cfg._instance.flat_out:
                ends = rows.row_ends.long()
                offs = torch.cumsum(ends, 0) - ends
                total = int(ends.sum())
                flat = torch.full((total + 256,), float("nan"))  # the tail is a guard
                launch(case, cfg, flat, schedule=schedule, out_offsets=offs)
            bad = []
            for i, ref in enumerate(refs):
                e = ref.numel()
                why = row_errors(out[i, :e], ref)
                if why is None and not bool(torch.isneginf(out[i, e:]).all()):
                    why = "written past its window"
                if why is None and flat is not None:
                    seg = flat[offs[i] : offs[i] + e]
                    if not torch.equal(
                        seg.view(torch.int32), out[i, :e].contiguous().view(torch.int32)
                    ):
                        why = "flat out differs from the plane"
                if why:
                    bad.append(f"row {i}: {why}")
            if not bool(
                torch.isneginf(store[:guard]).all()
                and torch.isneginf(store[-guard:]).all()
            ):
                bad.append("out's guard bands written")
            if flat is not None and not bool(torch.isnan(flat[total:]).all()):
                bad.append("the flat out's guard written")
            inst, mode = name.split(":")
            if mode == "table":
                table_out[inst] = out.clone()
            elif not torch.equal(
                out.view(torch.int32), table_out[inst].view(torch.int32)
            ):
                bad.append(f"differs from {inst}:table bit for bit")
            results.append((f"{tag} {name}", bad))
        if not all(torch.equal(a, b) for a, b in zip(inputs, before)):
            results.append((tag, ["an input was written"]))
    return results


def edge_case(model):
    """Nine sequences, each carrying one source of NaN / Inf; the case, each sequence's rows and
    what it carries. Two rows a full-layer sequence, so the multi-row instances take it too.
    """
    heads, n = MODEL_HEADS[model], 9
    if model == "cand":
        rows = candidate_layer("cand edges", [3000] * n, [1] * n)
    else:
        rows = full_layer(
            f"{model} edges", [3000] * n, [2] * n, 4 if model == "v4" else 2
        )
    case = make_case(rows, heads, page64=model == "v4")
    d = case.data
    seq_rows = [
        list(range(int(rows.query_start_loc[b]), int(rows.query_start_loc[b + 1])))
        for b in range(n)
    ]
    kr = [rows.key_rows(rr[-1]) for rr in seq_rows]
    ieee_heads = [0, 8] + ([32, 40] if heads == 64 else [16, 24])
    what = []

    def keys(b, value, block=None, a=100, z=110):
        for k in kr[b][a:z]:
            if block is None:
                d.scales[k] = value
            else:
                d.scales[k, block] = value

    keys(0, 0xFF, block=2)
    what.append("key scale 0xFF")
    keys(1, 0xFF, block=2)
    for r in seq_rows[1]:
        d.q[r, ieee_heads, 32:48] = 0  # that block zero in the heads whose relu is IEEE
    what.append("key scale 0xFF, q block zero in the IEEE heads")
    keys(2, 0xFF, block=2)
    for r in seq_rows[2]:
        d.q[r, :, 32:48] = 0
    what.append("key scale 0xFF, q block zero in all heads")
    d.q_scale[seq_rows[3][-1], 5, 1] = 0xFF
    what.append("q scale 0xFF")
    d.weights[seq_rows[4][-1], 7] = float("nan")
    what.append("weight NaN")
    for r in seq_rows[5]:
        d.q_scale[r] = 0xFE  # 2^127: products overflow to +-Inf
    keys(5, 0xFE, a=0, z=64)
    what.append("scales 2^127, overflow")
    d.weights[seq_rows[6][-1], 3] = float("inf")
    what.append("weight +Inf")
    for r in seq_rows[7]:
        d.q_scale[r, 5] = 0xBF  # 2^64 x 2^64: one head's dot product overflows
    keys(7, 0xBF)
    what.append("one head overflows (head 5)")
    for r in seq_rows[8]:
        d.q_scale[r, 0] = 0xBF  # ... through head 0, an IEEE one
    keys(8, 0xBF)
    what.append("one head overflows (head 0)")
    return case, seq_rows, what


def check_edges(model):
    case, seq_rows, what = edge_case(model)
    rows = case.rows
    refs = [run_torch(rows, case.data, r) for r in range(rows.rows)]
    results = []
    for name, (cfg, build) in ways(case).items():
        out = torch.full((rows.rows, rows.width), float("-inf"))
        launch(case, cfg, out, schedule=schedule_for(cfg, case, build))
        bad = []
        for b, rr in enumerate(seq_rows):
            for r in rr:
                e = refs[r].numel()
                why = row_errors(out[r, :e], refs[r])
                if why is None and not bool(torch.isneginf(out[r, e:]).all()):
                    why = "written past its window"
                if why:
                    bad.append(f"row {r} ({what[b]}): {why}")
        results.append((f"{rows.label} {name}", bad))
    return results


def band_rows(model):
    if model == "cand":
        return candidate_layer("cand bands", [20000, 17000, 9000], [6, 6, 6])
    return full_layer(
        f"{model} bands", [3000, 1700, 2600], [300, 180, 260], 4 if model == "v4" else 2
    )


def check_bands(model, band=97):
    """A forward scored in bands of rows -- per-row tensors sliced, query_start_loc clamped --
    equals it scored whole, bit for bit."""
    rows = band_rows(model)
    case = make_case(rows, MODEL_HEADS[model], page64=model == "v4")
    results = []
    for name, (cfg, build) in ways(case).items():
        whole = torch.full((rows.rows, rows.width), float("-inf"))
        launch(case, cfg, whole, schedule=schedule_for(cfg, case, build))
        banded = torch.full_like(whole, float("-inf"))
        for start in range(0, rows.rows, band):
            count = min(band, rows.rows - start)
            qsl = (rows.query_start_loc.clamp(start, start + count) - start).to(
                torch.int32
            )
            ends = rows.row_ends[start : start + count]
            launch(
                case,
                cfg,
                banded[start : start + count],
                schedule=schedule_for(cfg, case, build, qsl, ends),
                qsl=qsl,
                ends=ends,
                rows=slice(start, start + count),
            )
        same = torch.equal(whole.view(torch.int32), banded.view(torch.int32))
        results.append(
            (f"{rows.label} {name}", [] if same else [f"bands of {band} rows differ"])
        )
    return results


def check_padding(model, pad_rows=7):
    """Rows past query_start_loc[-1] (a CUDAGraph bucket's padding, with garbage windows) and empty
    sequences around every real one: the padding untouched, every real row right."""
    if model == "cand":
        real = candidate_layer("cand padding", [9000, 5000, 7000], [6, 6, 6])
    else:
        real = full_layer(
            f"{model} padding", [3000, 2000, 2600], [4, 2, 3], 4 if model == "v4" else 2
        )
    qsl, qsl2, tab = real.query_start_loc.tolist(), [0], []
    for b in range(real.num_seqs):
        qsl2 += [qsl2[-1], qsl[b + 1]]  # an empty sequence, then the real one
        tab += [real.block_tables[b], real.block_tables[b]]
    qsl2 += [qsl2[-1]] * 3  # empty sequences at the end (a padded batch)
    tab += [real.block_tables[-1]] * 3
    ends = torch.cat(
        [real.row_ends, torch.full((pad_rows,), real.width, dtype=torch.int32)]
    )
    rows = Rows(
        real.label,
        real.kind,
        torch.tensor(qsl2, dtype=torch.int32),
        ends,
        torch.stack(tab).contiguous(),
        real.pages_per_block,
        real.width,
        real.num_pages,
    )
    case = make_case(rows, MODEL_HEADS[model], page64=model == "v4")
    refs = [run_torch(real, case.data, r) for r in range(real.rows)]
    results = []
    for name, (cfg, build) in ways(case).items():
        out = torch.full((rows.rows, rows.width), float("-inf"))
        launch(case, cfg, out, schedule=schedule_for(cfg, case, build))
        bad = [
            f"row {r}: {w}"
            for r, ref in enumerate(refs)
            for w in [row_errors(out[r, : ref.numel()], ref)]
            if w
        ]
        if not bool(torch.isneginf(out[real.rows :]).all()):
            bad.append("a padding row was written")
        results.append((f"{rows.label} {name}", bad))
    return results


def check_graph(model):
    """Schedule + launch captured once, replayed on new contents of the same tensors (new q, new
    and shorter windows) == the same call made eagerly."""
    if model == "cand":
        rows = candidate_layer("cand graph", [20000, 17000], [6, 6])
    else:
        rows = full_layer(
            f"{model} graph", [9000, 1500, 777], [4, 3, 1], 4 if model == "v4" else 2
        )
    heads, page64 = MODEL_HEADS[model], model == "v4"
    results = []
    for name, (cfg, build) in ways(make_case(rows, heads, page64=page64)).items():
        case = make_case(rows, heads, page64=page64)
        other = make_case(rows, heads, page64=page64, seed=7)
        buf = cfg.schedule_buffers() if build else None
        out = torch.full((rows.rows, rows.width), float("-inf"))
        ends = rows.row_ends.clone()

        def step(cfg=cfg, case=case, out=out, ends=ends, buf=buf, build=build):
            s = (
                mqa_logits_mxfp4_schedule(cfg, rows.query_start_loc, ends, buffers=buf)
                if build
                else None
            )
            launch(case, cfg, out, schedule=s, ends=ends)

        step()  # warm: the module, the config's internal buffers
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        for src, dst in zip(
            (other.data.q, other.data.q_scale, other.data.weights),
            (case.data.q, case.data.q_scale, case.data.weights),
        ):
            dst.copy_(src)
        case.refresh()
        ends.copy_((rows.row_ends - 5).clamp_min(0))
        out.fill_(float("-inf"))
        g.replay()
        want = torch.full_like(out, float("-inf"))
        launch(
            case,
            cfg,
            want,
            schedule=schedule_for(cfg, case, build, ends=ends),
            ends=ends,
        )
        same = torch.equal(out.view(torch.int32), want.view(torch.int32))
        results.append(
            (f"{rows.label} {name}", [] if same else ["replay differs from eager"])
        )
    return results


def check_errors(model):
    """What the interface must refuse."""
    rows = model_rows(model, 2, f"{model} errors", [3000, 2000], [4, 4])
    case = make_case(rows, MODEL_HEADS[model], page64=model == "v4")
    cfg, d = config(case), case.data
    qsl, ends = rows.query_start_loc, rows.row_ends
    sched = mqa_logits_mxfp4_schedule(cfg, qsl, ends)

    def call(**kw):
        a = {
            "q": d.q,
            "q_scale": case.q_scale,
            "kv_cache": case.kv,
            "kv_scale": case.kv_scale,
            "weights": d.weights,
            "block_tables": case.block_tables,
            "query_start_loc": qsl,
            "row_ends": ends,
        }
        extra = {k: kw.pop(k) for k in ("schedule", "out", "out_offsets") if k in kw}
        a.update(kw)
        return lambda: mqa_logits_mxfp4(*a.values(), cfg=cfg, **extra)

    tries = {
        **(
            {"3-D (gfx1250) kv_scale": call(kv_scale=case.kv_scale.view(-1, 64, 4))}
            if _permuted()
            else {
                "4-D (gfx950) kv_scale": call(kv_scale=case.kv_scale.view(-1, 2, 32, 4))
            }
        ),
        "heads != cfg.heads": call(
            q=d.q[:, :16], q_scale=case.q_scale[:, :16], weights=d.weights[:, :16]
        ),
        "table narrower than cfg.block_table_width": call(
            block_tables=case.block_tables[:, :1].contiguous()
        ),
        "a schedule built from other tensors": call(
            query_start_loc=qsl.clone(), schedule=sched
        ),
        "rows over max_rows": call(
            q=torch.cat([d.q, d.q]),
            q_scale=torch.cat([case.q_scale, case.q_scale]),
            weights=torch.cat([d.weights, d.weights]),
            row_ends=torch.cat([ends, ends]),
        ),
        "int64 row_ends": call(row_ends=ends.long()),
        "table entries no instance reads": lambda: mqa_logits_mxfp4_config(
            torch.cuda.current_device(),
            heads=64,
            page_size=16,
            max_rows=64,
            max_seqs=64,
            max_qlen=4,
            max_seq_len=4096,
        ),
    }
    if not cfg._instance.flat_out:
        tries["out_offsets on an instance without a flat out"] = call(
            out=torch.empty(rows.rows * rows.width),
            out_offsets=torch.zeros(rows.rows, dtype=torch.int64),
        )
    results = []
    for what, f in tries.items():
        try:
            f()
            torch.cuda.synchronize()
            results.append((f"{rows.label} {what}", ["accepted"]))
        except (ValueError, RuntimeError):
            results.append((f"{rows.label} {what}", []))
    return results


SEMANTIC_CHECKS = {
    "rows": check_rows,
    "edges": check_edges,
    "bands": check_bands,
    "padding": check_padding,
    "graph": check_graph,
    "errors": check_errors,
}


@benchmark()
def test_semantics(model, check):
    results = SEMANTIC_CHECKS[check](model)
    torch.cuda.synchronize()
    failed = [(label, bad) for label, bad in results if bad]
    for label, bad in failed:
        aiter.logger.error("%s %s: %s", check, label, "; ".join(bad[:3]))
    return {
        "gfx": get_gfx(),
        "cases": len(results),
        "failed": len(failed),
        "result": "FAIL" if failed else "PASS",
        "first failure": f"{failed[0][0]}: {failed[0][1][0]}" if failed else "",
    }


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))
    return df


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mqa_logits_mxfp4 has no instances on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["decode", "prefill", "semantics"],
        choices=["decode", "prefill", "semantics"],
        help="which tables",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="*",
        default=["v4", "v41", "cand"],
        choices=list(MODEL_HEADS),
        help="v4 (64 heads), v41 / cand (32 heads)",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        nargs="*",
        default=[2, 1],
        help="V4.1 full layers' compression ratio (V4 is 4; cand ignores it)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128],
        help="decode: sequences",
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        nargs="*",
        default=[65536],
        help="context tokens a sequence (decode and prefill)",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        nargs="*",
        default=[1, 4, 6],
        help="decode: rows a sequence (MTP depth + 1)",
    )
    parser.add_argument(
        "-q",
        "--qlen",
        type=int,
        nargs="*",
        default=[16384, 4096],
        help="prefill: chunk rows a sequence (16384 rows in all)",
    )
    parser.add_argument(
        "--checks",
        nargs="*",
        default=list(SEMANTIC_CHECKS),
        choices=list(SEMANTIC_CHECKS),
        help="semantics: which checks",
    )
    args = parser.parse_args()
    for model in [m for m in args.model if not serves(m)]:
        aiter.logger.warning(
            "%s: %s", model, M._ELSEWHERE[get_gfx(), MODEL_HEADS[model]]
        )
    args.model = [m for m in args.model if serves(m)]

    def ratios(model):
        return [4] if model == "v4" else [1] if model == "cand" else args.ratio

    failed = False
    for model in args.model:
        if "decode" in args.modes:
            df = summarize(
                f"mqa_logits_mxfp4 {model} decode",
                [
                    test_decode(model, ratio, batch, ctx, mtp)
                    for ratio, batch, ctx, mtp in itertools.product(
                        ratios(model), args.batch, args.ctx, args.mtp
                    )
                ],
            )
            failed |= bool((df.filter(like=" err") > 0).any().any())
        if "prefill" in args.modes:
            df = summarize(
                f"mqa_logits_mxfp4 {model} prefill",
                [
                    test_prefill(model, ratio, qlen, ctx)
                    for ratio, qlen, ctx in itertools.product(
                        ratios(model), args.qlen, args.ctx
                    )
                ],
            )
            failed |= bool((df.filter(like=" err") > 0).any().any())
    if "semantics" in args.modes:
        df = summarize(
            "mqa_logits_mxfp4 semantics",
            [
                test_semantics(model, check)
                for model, check in itertools.product(args.model, args.checks)
            ],
        )
        failed |= bool((~df["result"].str.startswith("PASS")).any())
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
