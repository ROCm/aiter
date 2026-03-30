#!/usr/bin/env python3
"""
gen_perfetto.py

Converts a wave trace file (e.g., wgp0.txt) into a JSON file
compatible with Perfetto UI (https://ui.perfetto.dev/).

Usage:
  python3 gen_perfetto.py wgp0.txt out.json

Timestamps are zero-based cycle counts (minTS subtracted so the trace
starts at 0).  The original cycle number is preserved in each event's
args as "ts_raw".
Open the result at https://ui.perfetto.dev/ by dropping the file.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared helpers (originally from gen_timeline.py)
# ---------------------------------------------------------------------------

TS_RE = re.compile(r"\bTS=(\d+)\b")
WAVE_RE = re.compile(r"\[([^\]]*SIMD\d+_WAVE\d+)\]")


@dataclass
class Rec:
    wave: str
    ts: int
    instr: str
    cyc: int = 0


def strip_prefix_and_comments(disasm_line: str) -> str:
    colon = disasm_line.find(":")
    if colon != -1:
        disasm_line = disasm_line[colon + 1 :]
    disasm_line = disasm_line.strip()
    if "//" in disasm_line:
        disasm_line = disasm_line.split("//", 1)[0].rstrip()
    return " ".join(disasm_line.split())


def instr_class(instr: str) -> str:
    if not instr:
        return "other"
    mnemonic = instr.split()[0]

    if mnemonic.startswith("s_endpgm"):
        return "other"
    if mnemonic.startswith("s_barrier_wait"):
        return "s_wait"
    if mnemonic.startswith("s_wait"):
        return "s_wait"
    if mnemonic.startswith("s_"):
        return "s"

    if mnemonic.startswith("v_wmma"):
        return "vwmma"
    if mnemonic.startswith("v_"):
        return "v"

    if mnemonic.startswith("ds_"):
        return "ds"
    if mnemonic.startswith("tensor"):
        return "tensor"

    return "other"


def extract_simd(wave: str) -> str:
    m = re.search(r"(SIMD\d+)", wave)
    return m.group(1) if m else "SIMD??"


def simd_sort_key(simd: str):
    m = re.search(r"SIMD(\d+)", simd)
    return int(m.group(1)) if m else 10**9


def wave_sort_key(w: str):
    m = re.search(r"SIMD(\d+)_WAVE(\d+)", w)
    if not m:
        return (10**9, 10**9, w)
    return (int(m.group(1)), int(m.group(2)), w)


def parse_records(lines: list[str]) -> list[Rec]:
    recs: list[Rec] = []
    pending_wave: str | None = None
    pending_ts: int | None = None
    pending_disasm: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            pending_wave = None
            pending_ts = None
            pending_disasm = None
            continue

        if pending_ts is None:
            m_ts = TS_RE.search(line)
            m_wave = WAVE_RE.search(line)
            if not m_ts or not m_wave:
                continue
            pending_ts = int(m_ts.group(1))
            pending_wave = m_wave.group(1)
            continue

        if pending_disasm is None:
            instr = strip_prefix_and_comments(line)
            if instr:
                pending_disasm = instr

        if pending_wave is not None and pending_ts is not None and pending_disasm is not None:
            recs.append(Rec(wave=pending_wave, ts=pending_ts, instr=pending_disasm))
            pending_wave = None
            pending_ts = None
            pending_disasm = None

    return recs


def compute_cycles_per_wave(recs: list[Rec]) -> None:
    by_wave: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        by_wave[r.wave].append(r)

    for lst in by_wave.values():
        lst.sort(key=lambda x: x.ts)
        for i, r in enumerate(lst):
            if i + 1 < len(lst):
                r.cyc = max(0, lst[i + 1].ts - r.ts)
            else:
                r.cyc = 1


def build_trace(recs: list[Rec]) -> dict:
    compute_cycles_per_wave(recs)

    min_ts = min(r.ts for r in recs)

    # Group waves by SIMD.  In Perfetto's Chrome-trace importer a
    # "process" becomes a collapsible group and each "thread" inside
    # it becomes a track row.  Using large PID/TID values avoids
    # clashing with Perfetto-internal IDs.
    by_simd: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        by_simd[extract_simd(r.wave)].add(r.wave)

    simd_pid: dict[str, int] = {}
    wave_tid: dict[str, int] = {}
    meta_events: list[dict] = []

    for i, simd in enumerate(sorted(by_simd, key=simd_sort_key)):
        pid = 1000 + i
        simd_pid[simd] = pid
        meta_events.append({
            "name": "process_name", "ph": "M",
            "pid": pid, "tid": 0,
            "args": {"name": simd},
        })
        meta_events.append({
            "name": "process_sort_index", "ph": "M",
            "pid": pid, "tid": 0,
            "args": {"sort_index": i},
        })
        for j, wave in enumerate(sorted(by_simd[simd], key=wave_sort_key)):
            tid = j + 1       # avoid tid 0 (used by process metadata)
            wave_tid[wave] = tid
            meta_events.append({
                "name": "thread_name", "ph": "M",
                "pid": pid, "tid": tid,
                "args": {"name": wave},
            })
            meta_events.append({
                "name": "thread_sort_index", "ph": "M",
                "pid": pid, "tid": tid,
                "args": {"sort_index": j},
            })

    instr_events: list[dict] = []
    for r in recs:
        simd = extract_simd(r.wave)
        cls = instr_class(r.instr)
        mnemonic = r.instr.split()[0] if r.instr else "??"
        instr_events.append({
            "name": mnemonic,
            "cat": cls,
            "ph": "X",
            "ts": r.ts - min_ts,
            "dur": max(1, r.cyc),
            "pid": simd_pid[simd],
            "tid": wave_tid[r.wave],
            "args": {
                "instruction": r.instr,
                "wave": r.wave,
                "class": cls,
                "cycles": r.cyc,
                "ts_raw": r.ts,
            },
        })

    instr_events.sort(key=lambda e: e["ts"])
    return {"traceEvents": meta_events + instr_events}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 gen_perfetto.py trace.in out.json", file=sys.stderr)
        return 2

    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    recs = parse_records(lines)
    if not recs:
        print("No records parsed.", file=sys.stderr)
        return 1

    trace = build_trace(recs)
    n_instr = sum(1 for e in trace["traceEvents"] if e["ph"] == "X")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, separators=(",", ":"))

    print(f"Wrote {n_instr} events across {len(set(r.wave for r in recs))} waves to {out_path}")
    print(f"Open at: https://ui.perfetto.dev/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
