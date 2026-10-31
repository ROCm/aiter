# SPDX-License-Identifier: MIT
"""CPU-only *address audit* of the page-4 QSA logits kernel (no GPU, no flydsl).

The v0 kernel wedged the device on its first dispatch (dmesg:
``VM_PAGE_FAULT size=0x1000 start=0x16bf5fff000``).  This script runs nothing: it
re-derives, in plain Python, **every global and Q LDS address** the scalar kernel
can compute for a shape and checks it against the extent of its plane.  The fp8
cache path stages packed bf16 Q with scalar global loads and predicated LDS
stores, then consumes aligned 16-byte LDS chunks; bf16 keeps global vector loads.
The round-C bf16 WMMA address audit and its gate cases are retained separately.
This is the offline half of "this kernel cannot fault the device":

  * ``analytic`` — closed-form bounds.  Every address expression is monotone in
    ``(row, col, phys, off, s)``, so its maximum over the whole domain is the
    expression evaluated at the domain corner; that covers *all* workgroups,
    not a sample.
  * ``enumerate`` — per-column enumeration, including the tail lanes
    ``col >= columns`` (exhaustive for small shapes; boundary columns over
    several rows for the big buckets).  Only stores whose predicate is true are
    counted, so the audited set is exactly what the GPU would issue.
  * every offset is also checked against int32, because the kernel's
    ``buffer_*`` byte offsets are 32-bit;
  * the **v0 sentinel variant** is audited in parallel: v0 stored out-of-range
    lanes at the constant byte offset ``0x7FFFFFFF``.  The audit reports, per
    plane, whether the V# bounds check would have had to drop that access —
    and ``fault_check()`` re-derives the d07-2 fault address from such a store.

Usage::

    python audit_addressing_cpu.py                 # analytic + enumerate
    python audit_addressing_cpu.py --variant v0    # also audit v0's sentinel stores
    python audit_addressing_cpu.py --json out.json

Exit code 0 iff every audited access is inside its plane and fits int32.
"""

from __future__ import annotations

import argparse
import json
import sys

PAGE = 4          # compressed rows per index-cache page
HEADS = 4         # index heads
HDIM = 128        # head dim
THREADS = 256     # one thread per output column
Q_ROW_BYTES = HEADS * HDIM * 2          # 1024 B
Q_DWORDS = Q_ROW_BYTES // 4             # packed bf16 dwords staged for fp8
STEPS = HDIM // 8                       # 16 unrolled head-dim steps
MAX_I32 = 2**31 - 1
V0_SENTINEL = 0x7FFFFFFF                # v0's "dropped" store byte offset

DMESG_FAULT_PAGE = 0x16BF5FFF000        # d07-2, first dispatch attempt
DMESG_FAULT_SIZE = 0x1000


# ---------------------------------------------------------------------------
# Shape model (same formulas as the kernel / wrapper)
# ---------------------------------------------------------------------------
def columns_of(width, max_columns):
    full = width * PAGE
    return full if max_columns is None else min(full, int(max_columns))


def plane_sizes(rows, requests, width, cache_pages, fmt, columns):
    kv_elem = 1 if fmt == "fp8" else 2
    return {
        # LDS is one 1 KiB row per workgroup, not rows * 1 KiB.  bf16 does not
        # allocate it and its existing plane extents stay unchanged.
        **({"q_lds": Q_ROW_BYTES} if fmt == "fp8" else {}),
        "q": rows * Q_ROW_BYTES,
        "compressed_k_cache": cache_pages * PAGE * HDIM * kv_elem,
        "compressed_k_scale": cache_pages * PAGE * 4 if fmt == "fp8" else 4,
        "page_table": requests * width * 4,
        "token_to_request": rows * 4,
        "query_positions": rows * 4,
        "context_lens": requests * 4,
        "logits": rows * columns * 4,
        "visible_groups": rows * 4,
        "row_starts": rows * 4,
    }


def edge_page_table(requests, width, cache_pages):
    """A page table that hits every page-validity edge the mask must survive."""
    table = []
    for r in range(requests):
        row = []
        for c in range(width):
            if c == 0:
                row.append(-1)                       # invalid sentinel
            elif c == 1:
                row.append(cache_pages)              # OOB high (== num_pages)
            elif c == 2:
                row.append(cache_pages + 0x1000)     # far OOB high
            elif c == 3:
                row.append(cache_pages - 1)          # last valid page
            elif c == 4:
                row.append(0)                        # first valid page
            else:
                row.append((r * width + c) % max(1, cache_pages))
        table.append(row)
    return table


class Checker:
    """Accumulates audited accesses and reports those that leave their plane."""

    def __init__(self, sizes):
        self.sizes = sizes
        self.oob = []
        self.i32 = []
        self.n = 0
        self.n_stores = 0
        self.n_by_plane = {}
        self.n_stores_by_plane = {}
        self.unaligned = []
        self.max_end = {}

    def acc(self, plane, off, nbytes, detail="", alignment=1):
        self.n += 1
        self.n_by_plane[plane] = self.n_by_plane.get(plane, 0) + 1
        if off % alignment and len(self.unaligned) < 20:
            self.unaligned.append((plane, off, alignment, detail))
        end = off + nbytes
        size = self.sizes[plane]
        prev = self.max_end.get(plane)
        if prev is None or end > prev[0]:
            self.max_end[plane] = (end, detail)
        if off < 0 or end > size:
            if len(self.oob) < 40:
                self.oob.append((plane, off, nbytes, size, detail))
        if end > MAX_I32:
            if len(self.i32) < 20:
                self.i32.append((plane, off, nbytes, detail))

    def store(self, plane, off, nbytes, detail="", alignment=1):
        self.n_stores += 1
        self.n_stores_by_plane[plane] = self.n_stores_by_plane.get(plane, 0) + 1
        self.acc(plane, off, nbytes, detail, alignment)


def audit_analytic(rows, requests, width, cache_pages, fmt, max_columns=None):
    """Closed form: every address expression is monotone, so the corner is worst."""
    columns = columns_of(width, max_columns)
    sizes = plane_sizes(rows, requests, width, cache_pages, fmt, columns)
    kv_elem = 1 if fmt == "fp8" else 2
    page_bytes = PAGE * HDIM * kv_elem
    key_row_bytes = HDIM * kv_elem
    k_bytes = (2 if fmt == "fp8" else 4) * 4
    c = Checker(sizes)
    last_row, last_col = rows - 1, columns - 1

    # --- loads: each bound is the corner of a masked index -------------------
    c.acc("token_to_request", last_row * 4, 4, "row <= rows-1")
    c.acc("query_positions", last_row * 4, 4, "row <= rows-1")
    c.acc("context_lens", (requests - 1) * 4, 4, "safe_req <= num_requests-1")
    c.acc("page_table", ((requests - 1) * width + (width - 1)) * 4, 4,
          "safe_req, safe_logical at max")
    if fmt == "fp8":
        # Every lane loads a clamped dword, even if the following store is not
        # taken.  Surplus lanes therefore repeat this last safe scalar address.
        c.acc("q", last_row * Q_ROW_BYTES + (Q_DWORDS - 1) * 4, 4,
              "row, safe_q_dw at max (scalar stage load)", alignment=4)
        c.store("q_lds", (Q_DWORDS - 1) * 4, 4,
                "q_dw < q_dwords; safe_q_dw at max", alignment=4)
        c.acc("q_lds", (HEADS - 1) * HDIM * 2 + (STEPS - 1) * 16, 16,
              "h, s at max (vector LDS read)", alignment=16)
        c.store("q_lds", 0, 4, "first staging writer", alignment=4)
        c.acc("q_lds", 0, 16, "first consumer chunk", alignment=16)
    else:
        c.acc("q", last_row * Q_ROW_BYTES + (HEADS - 1) * HDIM * 2 + (STEPS - 1) * 16, 16,
              "row, h, s at max")
    c.acc("compressed_k_cache",
          (cache_pages - 1) * page_bytes + (PAGE - 1) * key_row_bytes + (STEPS - 1) * k_bytes,
          k_bytes, "phys, off, s at max")
    if fmt == "fp8":
        c.acc("compressed_k_scale", ((cache_pages - 1) * PAGE + (PAGE - 1)) * 4, 4,
              "phys, off at max")

    # --- the min corner: every index is >= 0 after the masks, so 0 is the min
    for plane, width_bytes in (("q", 4 if fmt == "fp8" else 16), ("compressed_k_cache", k_bytes),
                               ("page_table", 4), ("token_to_request", 4),
                               ("query_positions", 4), ("context_lens", 4)):
        c.acc(plane, 0, width_bytes, "min corner (all indices clamped >= 0)")

    # --- stores: predicated, so the corner is the last row's last column -----
    c.store("visible_groups", last_row * 4, 4, "writer")
    c.store("row_starts", last_row * 4, 4, "writer")
    idx = min(last_row * columns + last_col, rows * columns - 1)
    if idx > MAX_I32:
        c.i32.append(("logits", idx, 0, "out_idx"))
    c.store("logits", idx * 4, 4, "row, col at max")

    return {
        "mode": "analytic",
        "sizes": sizes,
        "grid": {
            # grid.x = ceil(columns / threads), grid.y = rows; the grid must cover
            # every output element and must not exceed the y limit (65535).
            "threads": THREADS,
            "ptiles": (columns + THREADS - 1) // THREADS,
            "lanes": ((columns + THREADS - 1) // THREADS) * THREADS,
            "covers_all_columns": ((columns + THREADS - 1) // THREADS) * THREADS >= columns,
            "grid_y_ok": rows <= 65535,
        },
        "accesses": c.n,
        "stores": c.n_stores,
        "accesses_by_plane": c.n_by_plane,
        "stores_by_plane": c.n_stores_by_plane,
        "unaligned": c.unaligned,
        "oob": c.oob,
        "i32": c.i32,
        "max_offset_plus_width": {k: v[0] for k, v in c.max_end.items()},
        "ok": not c.oob and not c.i32 and not c.unaligned,
    }


def _visit_cols(columns, threads, full):
    ptiles = (columns + threads - 1) // threads
    lanes = ptiles * threads                 # every lane the grid launches
    if full:
        return list(range(lanes))
    edge = set(range(min(4, lanes)))
    edge |= set(range(max(0, lanes - 4), lanes))
    for col in (columns - 1, columns, columns + 1):
        if 0 <= col < lanes:
            edge.add(col)
    for t in range(1, ptiles):               # block boundaries
        for col in (t * threads - 1, t * threads):
            if 0 <= col < lanes:
                edge.add(col)
    return sorted(edge)


def audit_enumerate(rows, requests, width, cache_pages, fmt, max_columns=None,
                    threads=THREADS, full_limit=40_000, max_rows=8):
    columns = columns_of(width, max_columns)
    sizes = plane_sizes(rows, requests, width, cache_pages, fmt, columns)
    kv_elem = 1 if fmt == "fp8" else 2
    page_bytes = PAGE * HDIM * kv_elem
    key_row_bytes = HDIM * kv_elem
    k_bytes = (2 if fmt == "fp8" else 4) * 4
    table = edge_page_table(requests, width, cache_pages)
    lanes = ((columns + threads - 1) // threads) * threads
    full = rows * lanes <= full_limit
    if full:
        row_iter = list(range(rows))
    else:
        cand = sorted({0, 1, rows - 2, rows - 1})
        row_iter = [r for r in cand if 0 <= r < rows][:max_rows]
    cols = _visit_cols(columns, threads, full)
    c = Checker(sizes)
    sentinel = 0
    ptiles = sorted({col // threads for col in cols})
    for row in row_iter:
        if fmt == "fp8":
            # Staging is block-wide and precedes metadata/output predicates.
            # Even boundary-only audits enumerate *all* staging lanes of each
            # represented block, including surplus lanes whose store is skipped.
            for ptile in ptiles:
                for qi in range((Q_DWORDS + threads - 1) // threads):
                    for tid in range(threads):
                        q_dw = tid + qi * threads
                        safe_q_dw = min(q_dw, Q_DWORDS - 1)
                        detail = f"row={row} ptile={ptile} tid={tid} qi={qi} q_dw={q_dw}"
                        c.acc("q", row * Q_ROW_BYTES + safe_q_dw * 4, 4,
                              detail + " (clamped scalar stage load)", alignment=4)
                        if q_dw < Q_DWORDS:
                            c.store("q_lds", safe_q_dw * 4, 4,
                                    detail + " (predicated stage store)", alignment=4)
        c.acc("token_to_request", row * 4, 4, f"row={row}")
        c.acc("query_positions", row * 4, 4, f"row={row}")
        for col in cols:
            col_ok = col < columns
            logical, off = col // PAGE, col % PAGE
            logical_ok = logical < width
            safe_logical = logical if logical_ok else 0
            for req in sorted({0, requests - 1}):
                c.acc("context_lens", req * 4, 4, f"row={row} req={req}")
                c.acc("page_table", (req * width + safe_logical) * 4, 4,
                      f"req={req} logical={safe_logical} (masked)")
                entry = table[req][safe_logical]
                page_ok = 0 <= entry < cache_pages
                phys = entry if page_ok else 0
                k_base = phys * page_bytes + off * key_row_bytes
                for s in range(STEPS):
                    c.acc("compressed_k_cache", k_base + s * k_bytes, k_bytes,
                          f"phys={phys} off={off} s={s} page_ok={page_ok}")
                if fmt == "fp8":
                    c.acc("compressed_k_scale", (phys * PAGE + off) * 4, 4,
                          f"phys={phys} off={off} page_ok={page_ok}")
            for h in range(HEADS):
                for s in range(STEPS):
                    if fmt == "fp8":
                        # No row term: q_ptr addresses this workgroup's LDS row.
                        c.acc("q_lds", (h * (HDIM // 2) + s * 4) * 4, 16,
                              f"row={row} col={col} h={h} s={s}", alignment=16)
                    else:
                        c.acc("q", row * Q_ROW_BYTES + h * HDIM * 2 + s * 16, 16,
                              f"row={row} h={h} s={s}")
            if col_ok:
                idx = min(row * columns + col, rows * columns - 1)
                c.store("logits", idx * 4, 4, f"row={row} col={col}")
            else:
                sentinel += 1
            if col == 0:
                c.store("visible_groups", row * 4, 4, "writer (tid==0, ptile==0)")
                c.store("row_starts", row * 4, 4, "writer (tid==0, ptile==0)")
            else:
                sentinel += 2
    return {
        "mode": "enumerate",
        "full": full,
        "rows_visited": len(row_iter),
        "cols_visited": len(cols),
        "lanes": lanes,
        "sizes": sizes,
        "accesses": c.n,
        "stores": c.n_stores,
        "accesses_by_plane": c.n_by_plane,
        "stores_by_plane": c.n_stores_by_plane,
        "unaligned": c.unaligned,
        "oob": c.oob,
        "i32": c.i32,
        "max_offset_plus_width": {k: v[0] for k, v in c.max_end.items()},
        "sentinel_events": sentinel,
        "sentinel_dropped_if_armed": all(
            V0_SENTINEL >= max(sizes[p], 1) for p in ("logits", "visible_groups", "row_starts")),
        "ok": not c.oob and not c.i32 and not c.unaligned,
    }


# ---------------------------------------------------------------------------
# bf16 WMMA path (round C): one workgroup = 4 waves x 32 lanes = 16 rows x 64 cols
# ---------------------------------------------------------------------------
WM_WAVE, WM_WAVES, WM_ROWS, WM_N, WM_K, WM_NT = 32, 4, 4, 16, 32, 4
WM_COLS = WM_N * WM_NT
WM_KS = HDIM // WM_K


def _wm_lane_addrs(rows, requests, width, cache_pages, columns, table, by, bx, wave, lane):
    """Every global address one lane of the WMMA kernel can issue (taken or not)."""
    n_in, kgrp = lane % WM_N, lane // WM_N
    t0 = (by * WM_WAVES + wave) * WM_ROWS
    col0 = bx * WM_COLS
    acc = []                                     # (plane, off, nbytes, is_store, i32_terms)
    for i in range(WM_ROWS):                     # row metadata (clamped rows)
        safe_row = min(t0 + i, rows - 1)
        acc += [("token_to_request", safe_row * 4, 4, False, ()),
                ("query_positions", safe_row * 4, 4, False, ())]
        for req in (0, requests - 1):            # safe_req range
            acc.append(("context_lens", req * 4, 4, False, ()))
    my_row = t0 + min(lane, WM_ROWS - 1)         # metadata store (clamped)
    acc += [("visible_groups", min(my_row, rows - 1) * 4, 4, True, ()),
            ("row_starts", min(my_row, rows - 1) * 4, 4, True, ())]
    q_row = min(t0 + n_in // HEADS, rows - 1)    # Q fragments
    q_base = q_row * Q_ROW_BYTES + (n_in % HEADS) * HDIM * 2 + kgrp * 16
    for s in range(WM_KS):
        acc += [("q", q_base + s * 64, 16, False, ()), ("q", q_base + s * 64 + 32, 16, False, ())]
    last = rows * columns - 1
    row_a, row_b = min(t0 + 2 * kgrp, rows - 1), min(t0 + 2 * kgrp + 1, rows - 1)
    for nt in range(WM_NT):
        c = col0 + nt * WM_N + n_in
        logical, off = c // PAGE, c % PAGE
        safe_logical = logical if logical < width else 0
        for req in (0, requests - 1):
            acc.append(("page_table", (req * width + safe_logical) * 4, 4, False, ()))
            entry = table[req][safe_logical]
            phys = entry if 0 <= entry < cache_pages else 0
            k_base = phys * PAGE * HDIM * 2 + off * HDIM * 2 + kgrp * 16
            for s in range(WM_KS):
                acc += [("compressed_k_cache", k_base + s * 64, 16, False, ()),
                        ("compressed_k_cache", k_base + s * 64 + 32, 16, False, ())]
        for r in (row_a, row_b):
            acc.append(("logits", min(r * columns + c, last) * 4, 4, True, (r * columns + c,)))
    return acc


def audit_wmma(rows, requests, width, cache_pages, max_columns=None, full_limit=4096):
    """Audit every lane of every workgroup (small shapes) or of the boundary
    workgroups (big shapes: first/last row block x first/last two column blocks)."""
    columns = columns_of(width, max_columns)
    sizes = plane_sizes(rows, requests, width, cache_pages, "bf16", columns)
    table = edge_page_table(requests, width, cache_pages)
    gx = (columns + WM_COLS - 1) // WM_COLS
    gy = (rows + WM_ROWS * WM_WAVES - 1) // (WM_ROWS * WM_WAVES)
    full = gx * gy <= full_limit
    bys = range(gy) if full else sorted({0, gy - 1})
    bxs = range(gx) if full else sorted({0, 1, max(0, gx - 2), gx - 1})
    c = Checker(sizes)
    for by in bys:
        for bx in bxs:
            for wave in range(WM_WAVES):
                for lane in range(WM_WAVE):
                    for plane, off, nb, st, terms in _wm_lane_addrs(
                            rows, requests, width, cache_pages, columns, table, by, bx, wave, lane):
                        (c.store if st else c.acc)(plane, off, nb, f"by={by} bx={bx} w={wave} l={lane}")
                        for t in terms:
                            if t > MAX_I32:
                                c.i32.append((plane, t, 0, "i32 intermediate"))
    return {
        "mode": "wmma",
        "full": full,
        "grid": {"x": gx, "y": gy, "covers_columns": gx * WM_COLS >= columns,
                 "covers_rows": gy * WM_ROWS * WM_WAVES >= rows, "grid_y_ok": gy <= 65535},
        "accesses": c.n,
        "stores": c.n_stores,
        "oob": c.oob,
        "i32": c.i32,
        "max_offset_plus_width": {k: v[0] for k, v in c.max_end.items()},
        "ok": (not c.oob and not c.i32 and gx * WM_COLS >= columns
               and gy * WM_ROWS * WM_WAVES >= rows and gy <= 65535),
    }


# ---------------------------------------------------------------------------
# v0 fault forensics: is the dmesg address a sentinel store?
# ---------------------------------------------------------------------------
def fault_check():
    """Re-derive the d07-2 fault address from a sentinel store.

    A store at byte offset ``0x7FFFFFFF`` from a device allocation base ``B``
    faults at ``B + 0x7FFFFFFF``; dmesg reports the *page start* of the faulting
    address.  For a page-aligned ``B`` that page start is ``B + 0x7FFFF000``, so
    it must be ``0x1FF000`` mod 2 MiB when the allocation is 2 MiB aligned - and
    the implied ``B`` must be aligned.
    """
    page = DMESG_FAULT_PAGE
    implied = page - 0x7FFFF000
    return {
        "dmesg": {"size": hex(DMESG_FAULT_SIZE), "page_start": hex(page)},
        "sentinel_off": hex(V0_SENTINEL),
        "observed_page_start_mod_2MiB": hex(page % (2 * 1024 * 1024)),
        "required_mod_2MiB_if_sentinel": hex(0x7FFFF000 % (2 * 1024 * 1024)),
        "implied_allocation_base": hex(implied),
        "implied_base_mod_2MiB": hex(implied % (2 * 1024 * 1024)),
        "implied_base_mod_4KiB": implied % 4096,
        "fault_address_if_sentinel": hex(implied + V0_SENTINEL),
        "consistent_with_sentinel_store": (
            (page % (2 * 1024 * 1024)) == (0x7FFFF000 % (2 * 1024 * 1024))
            and implied % 4096 == 0
        ),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def cases():
    """(name, rows, requests, width, cache_pages, fmt, max_columns)."""
    return [
        # first-touch ladder (see FIRST_TOUCH.md)
        ("ft_a_1row_1req_4col_bf16", 1, 1, 1, 8, "bf16", None),
        ("ft_a_1row_1req_4col_fp8", 1, 1, 1, 8, "fp8", None),
        ("ft_b_1row_1req_8col_bf16", 1, 1, 2, 8, "bf16", None),
        ("ft_c_4rows_1req_16col_bf16", 4, 1, 4, 16, "bf16", None),
        ("ft_d_fp8_scale", 4, 1, 4, 16, "fp8", None),
        # the I5 smoke case that was first dispatched on d07-2
        ("I5_min_decode_bf16", 4, 1, 16, 64, "bf16", None),
        ("I5_min_decode_fp8", 4, 1, 16, 64, "fp8", None),
        # column tails: columns % 4 != 0 and non-multiples of the block
        ("tail_cols_1", 4, 1, 16, 64, "bf16", 1),
        ("tail_cols_3", 4, 1, 16, 64, "bf16", 3),
        ("tail_cols_5", 4, 1, 16, 64, "bf16", 5),
        ("tail_cols_255", 4, 1, 64, 256, "bf16", 255),
        ("tail_cols_257", 4, 1, 65, 256, "bf16", 257),
        ("tail_cols_1000_fp8", 8, 2, 256, 1024, "fp8", 1000),
        # one logical page only / few columns
        ("width_1_bf16", 4, 1, 1, 4, "bf16", None),
        ("rows_64_cols_4", 64, 2, 1, 8, "bf16", None),
        # contract buckets
        ("decode_64x65536_bf16", 64, 16, 16384, 738964, "bf16", None),
        ("prefill_512x65536_fp8", 512, 2, 16384, 738964, "fp8", None),
        ("I14_decode_c32768_bf16", 64, 16, 16384, 738964, "bf16", 32768),
        # round C (bf16 WMMA path): partial waves / workgroups, draft, prefill chunk,
        # and the i32 logits cap (rows * columns * 4 = 2**31 - 2**18)
        ("wm_rows7_cols70_bf16", 7, 2, 32, 128, "bf16", 70),
        ("wm_rows13_bf16", 13, 2, 32, 128, "bf16", None),
        ("wm_rows70_bf16", 70, 2, 16, 64, "bf16", None),
        ("wm_draft_16x65536_bf16", 16, 16, 16384, 738964, "bf16", None),
        ("wm_prefill_512x65536_bf16", 512, 2, 16384, 738964, "bf16", None),
        ("wm_i32_cap_8191x65536_bf16", 8191, 2, 16384, 738964, "bf16", None),
        # round F fp8 LDS path: tail blocks participate fully in staging/barrier.
        ("fp8_lds_tail_cols_1", 1, 1, 1, 4, "fp8", 1),
        ("fp8_lds_tail_cols_257", 4, 2, 65, 256, "fp8", 257),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="current", choices=["current", "v0", "both"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--max-bytes", type=int, default=MAX_I32,
                    help="i32 addressing cap the op wrapper enforces")
    args = ap.parse_args(argv)

    out = {"fault_check": fault_check(), "cases": []}
    rc = 0
    if not args.quiet:
        fc = out["fault_check"]
        print("== v0 fault forensics (d07-2 dmesg) ==")
        print("  observed fault page start : %s size=%s"
              % (fc["dmesg"]["page_start"], fc["dmesg"]["size"]))
        print("  page_start mod 2 MiB      : %s   required for a 0x7FFFFFFF store: %s"
              % (fc["observed_page_start_mod_2MiB"], fc["required_mod_2MiB_if_sentinel"]))
        print("  implied allocation base   : %s (mod 2 MiB %s, mod 4 KiB %d)"
              % (fc["implied_allocation_base"], fc["implied_base_mod_2MiB"],
                 fc["implied_base_mod_4KiB"]))
        print("  fault addr if sentinel    : %s" % fc["fault_address_if_sentinel"])
        print("  consistent with sentinel  : %s" % fc["consistent_with_sentinel_store"])
        print("\n== address audit (current kernel: predicated stores, i64 extents, cap %d B) =="
              % args.max_bytes)

    for (name, rows, requests, width, cache_pages, fmt, max_columns) in cases():
        a = audit_analytic(rows, requests, width, cache_pages, fmt, max_columns)
        e = audit_enumerate(rows, requests, width, cache_pages, fmt, max_columns)
        oversized = {k: v for k, v in a["sizes"].items() if v > args.max_bytes}
        # bf16 is dispatched to the WMMA kernel (round C): audit that mapping too
        w = audit_wmma(rows, requests, width, cache_pages, max_columns) if fmt == "bf16" else None
        rec = {
            "case": name,
            "shape": {"rows": rows, "requests": requests, "width": width,
                      "columns": columns_of(width, max_columns),
                      "cache_pages": cache_pages, "fmt": fmt},
            "analytic": a,
            "enumerate": e,
            "wmma": w,
            "planes_over_cap": oversized,
            "ok": (a["ok"] and e["ok"] and not oversized
                   and a["grid"]["covers_all_columns"] and a["grid"]["grid_y_ok"]
                   and (w is None or w["ok"])),
        }
        out["cases"].append(rec)
        rc |= 0 if rec["ok"] else 1
        if not args.quiet:
            print("[%s] %-26s %-4s rows=%4d reqs=%3d cols=%6d pages=%7d | analytic oob=%d i32=%d"
                  " | enum oob=%d i32=%d rows=%d cols=%d%s"
                  % ("PASS" if rec["ok"] else "FAIL", name, fmt, rows, requests,
                     rec["shape"]["columns"], cache_pages, len(a["oob"]), len(a["i32"]),
                     len(e["oob"]), len(e["i32"]), e["rows_visited"], e["cols_visited"],
                     "" if rec["ok"] else "   <-- CHECK"))
            if w is not None:
                print("       wmma: grid=%dx%d %s accesses=%d stores=%d oob=%d i32=%d max_end=%s"
                      % (w["grid"]["x"], w["grid"]["y"], "full" if w["full"] else "boundary",
                         w["accesses"], w["stores"], len(w["oob"]), len(w["i32"]),
                         {k: v for k, v in w["max_offset_plus_width"].items()
                          if k in ("q", "compressed_k_cache", "logits")}))
                for (plane, off, nbytes, size, detail) in w["oob"][:3]:
                    print("        WMMA OOB %s off=%d+%d > %d (%s)" % (plane, off, nbytes, size, detail))
            if fmt == "fp8":
                print("       q LDS: extent=%d global scalar loads=%d LDS stores=%d "
                      "vector reads=%d unaligned=%d"
                      % (e["sizes"]["q_lds"], e["accesses_by_plane"]["q"],
                         e["stores_by_plane"]["q_lds"],
                         e["accesses_by_plane"]["q_lds"] - e["stores_by_plane"]["q_lds"],
                         len(a["unaligned"]) + len(e["unaligned"])))
            for issue in (a["unaligned"] + e["unaligned"])[:3]:
                print("        UNALIGNED %s off=%d alignment=%d (%s)" % issue)
            if oversized:
                print("        over i32 cap: %s" % oversized)
            for (plane, off, nbytes, size, detail) in (a["oob"] + e["oob"])[:3]:
                print("        OOB %s off=%d+%d > %d (%s)" % (plane, off, nbytes, size, detail))

    if args.variant in ("v0", "both"):
        if not args.quiet:
            print("\n== v0 (sentinel) store audit: out-of-range lanes stored at 0x%X =="
                  % V0_SENTINEL)
        for (name, rows, requests, width, cache_pages, fmt, max_columns) in cases():
            e = audit_enumerate(rows, requests, width, cache_pages, fmt, max_columns)
            out.setdefault("v0_sentinel", []).append({
                "case": name,
                "sentinel_events": e["sentinel_events"],
                "all_would_be_dropped_if_vsharp_armed": e["sentinel_dropped_if_armed"],
    })
            if not args.quiet:
                print("  %-26s sentinel stores=%8d  dropped_if_vsharp_armed=%s"
                      % (name, e["sentinel_events"], e["sentinel_dropped_if_armed"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        if not args.quiet:
            print("\nwrote", args.json)
    if not args.quiet:
        print("\naddress audit: %s" % ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
