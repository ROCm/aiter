#!/usr/bin/env python3
"""Markdown summary of an e2e.sh run directory: every arm against the first.

    e2e_summary.py <run dir>

The arms and their order come from <run dir>/arms.txt (the first is the
reference); without it they are "main" then the rest, alphabetically.
Prints up to two tables:
  - per (TP, scenario) median TTFT, median TPOT and output throughput of each
    arm from <tag>_tp<N>/bench_<scenario>.json, with the delta of every other
    arm against the first (positive = better than the first);
  - per TP and arm, the first token where its greedy smoke completion differs
    from the first arm's (<tag>_tp<N>/smoke.json), with each server's logprob
    margin of its own pick over the other's.
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


def arm_order(run_dir):
    """Arm tags, reference first."""
    f = run_dir / "arms.txt"
    if f.exists():
        return f.read_text().split()
    tags = {re.match(r"(\w+)_tp\d+", d.name).group(1) for d in run_dir.glob("*_tp*")}
    return sorted(tags, key=lambda t: (t != "main", t))


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


def bench_table(rows, arms):
    ref, others = arms[0], arms[1:]
    head = ["TP", "scenario"]
    for _, name, _ in METRICS:
        head += ["%s %s" % (name, t) for t in arms]
        head += ["%s: %s vs %s" % (name, t, ref) for t in others]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (tp, scenario), r in sorted(rows.items(), key=scenario_order):
        cells = [str(tp), scenario]
        for key, _, sign in METRICS:
            v = {t: r.get(t, {}).get(key) for t in arms}
            cells += ["%.1f" % v[t] if v[t] else "-" for t in arms]
            cells += [
                "%+.1f%%" % (sign * (v[t] - v[ref]) / v[ref] * 100)
                if v[ref] and v[t] else "-"
                for t in others
            ]
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


def smoke_table(run_dir, arms):
    ref, lines = arms[0], []
    for ref_dir in sorted(run_dir.glob(ref + "_tp*")):
        tp = ref_dir.name[len(ref + "_tp"):]
        a = smoke_logprobs(ref_dir / "smoke.json")
        for tag in arms[1:]:
            b = smoke_logprobs(run_dir / ("%s_tp%s" % (tag, tp)) / "smoke.json")
            if not (a and b):
                continue
            ta, tb = a["tokens"], b["tokens"]
            pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
            if pos is None:
                lines.append("| %s | %s | identical (%d tokens) | | | | |" % (tp, tag, len(ta)))
                continue
            lines.append("| %s | %s | %d | `%s` | `%s` | %s | %s |" % (
                tp, tag, pos, json.dumps(ta[pos]), json.dumps(tb[pos]),
                margin(a, pos, ta[pos], tb[pos]), margin(b, pos, tb[pos], ta[pos])))
    if not lines:
        return []
    return [
        "Greedy smoke test: first token where an arm differs from %s. Margin = "
        "logprob of the server's own pick minus logprob of the other's pick; "
        "a small margin on both sides means a near-tie broken by rounding." % ref,
        "",
        "| TP | arm | position | %s picks | arm picks | %s margin | arm margin |" % (ref, ref),
        "|---|---|---|---|---|---|---|",
    ] + lines


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    arms = arm_order(run_dir)
    if len(arms) < 1:
        return
    rows = load_bench(run_dir)
    sections = [
        s for s in (bench_table(rows, arms) if rows else [], smoke_table(run_dir, arms)) if s
    ]
    print("\n\n".join("\n".join(s) for s in sections))


if __name__ == "__main__":
    main()
