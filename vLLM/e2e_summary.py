#!/usr/bin/env python3
"""Markdown summary of an e2e.sh run directory: main (A) vs ar (B).

    e2e_summary.py <run dir>

Prints up to two tables:
  - per (TP, scenario) median TTFT, median TPOT and output throughput from
    <tag>_tp<N>/bench_<scenario>.json, with the delta of ar against main
    (positive = ar is better);
  - per TP, the first token where the greedy smoke completions of main and ar
    differ (<tag>_tp<N>/smoke.json), with each server's logprob margin of its
    own pick over the other's.
"""

import json
import re
import sys
from pathlib import Path

# (bench json key, column name, sign: -1 if lower is better)
METRICS = [
    ("median_ttft_ms", "TTFT ms", -1),
    ("median_tpot_ms", "TPOT ms", -1),
    ("output_throughput", "out tok/s", 1),
]


def load_bench(run_dir):
    """{(tp, scenario): {tag: bench json}}"""
    rows = {}
    for f in run_dir.glob("*_tp*/bench_*.json"):
        tag, tp = re.match(r"(\w+)_tp(\d+)", f.parent.name).groups()
        scenario = f.stem[len("bench_"):]
        rows.setdefault((int(tp), scenario), {})[tag] = json.loads(f.read_text())
    return rows


def scenario_order(item):
    (tp, scenario), _ = item
    shape, conc = scenario.split("_c")
    inp, out = shape.split("x")
    return (-tp, int(inp), int(out), int(conc))


def bench_table(rows):
    head = ["TP", "scenario"]
    for _, name, _ in METRICS:
        head += [name + " main", name + " ar", "delta"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (tp, scenario), r in sorted(rows.items(), key=scenario_order):
        cells = [str(tp), scenario]
        for key, _, sign in METRICS:
            a, b = r.get("main", {}).get(key), r.get("ar", {}).get(key)
            cells += ["%.1f" % a if a else "-", "%.1f" % b if b else "-"]
            cells.append("%+.1f%%" % (sign * (b - a) / a * 100) if a and b else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def smoke_logprobs(path):
    """The logprobs block of a smoke.json, or None if missing or not requested."""
    try:
        return json.loads(path.read_text())["choices"][0]["logprobs"]
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None


def margin(logprobs, pos, own, other):
    """Logprob of own pick minus the other server's pick at pos."""
    top = logprobs["top_logprobs"][pos]
    if other not in top:
        return "not in top-%d" % len(top)
    return "%.3f" % (top[own] - top[other])


def smoke_table(run_dir):
    lines = []
    for main_dir in sorted(run_dir.glob("main_tp*")):
        tp = main_dir.name[len("main_tp"):]
        a = smoke_logprobs(main_dir / "smoke.json")
        b = smoke_logprobs(run_dir / ("ar_tp" + tp) / "smoke.json")
        if not (a and b):
            continue
        ta, tb = a["tokens"], b["tokens"]
        pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if pos is None:
            lines.append("| %s | identical (%d tokens) | | | | |" % (tp, len(ta)))
            continue
        lines.append("| %s | %d | `%s` | `%s` | %s | %s |" % (
            tp, pos, json.dumps(ta[pos]), json.dumps(tb[pos]),
            margin(a, pos, ta[pos], tb[pos]), margin(b, pos, tb[pos], ta[pos])))
    if not lines:
        return []
    return [
        "Greedy smoke test: first token where main and ar differ. Margin = "
        "logprob of the server's own pick minus logprob of the other's pick; "
        "a small margin on both sides means a near-tie broken by rounding.",
        "",
        "| TP | position | main picks | ar picks | main margin | ar margin |",
        "|---|---|---|---|---|---|",
    ] + lines


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    rows = load_bench(run_dir)
    sections = [s for s in (bench_table(rows) if rows else [], smoke_table(run_dir)) if s]
    print("\n\n".join("\n".join(s) for s in sections))


if __name__ == "__main__":
    main()
