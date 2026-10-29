#!/usr/bin/env python3
"""Markdown summary of an `e2e.sh eval` run directory: every arm against the first.

    e2e_eval_summary.py <run dir>

The arms and their order come from <run dir>/arms.txt (the first is the
reference). Each arm's lm-eval output is under
<tag>_tp<N>/traces/lm_eval/<model>/{results_*.json,samples_<task>_*.jsonl};
if an arm has several, the newest is used. Prints two tables:
  - per (TP, task, metric) the score of each arm with its standard error, and
    the difference of every other arm from the first, with the standard error
    of the difference (the two runs taken as independent);
  - per (TP, task, metric) and arm, the paired comparison with the first arm on
    the same questions: how many each one alone got right, and the two-sided
    exact McNemar p-value of that split. Greedy decoding makes the pairing
    tight, so this is the more sensitive test.
"""

import json
import math
import re
import sys
from pathlib import Path


def arm_order(run_dir):
    """Arm tags, reference first."""
    f = run_dir / "arms.txt"
    if f.exists():
        return f.read_text().split()
    return sorted(
        {re.match(r"(\w+)_tp\d+", d.name).group(1) for d in run_dir.glob("*_tp*")}
    )


def newest(paths):
    return max(paths, key=lambda p: p.name) if paths else None


def load_arm(arm_dir):
    """(results json, {task: samples path}) of one arm, or None if it has none."""
    root = arm_dir / "traces" / "lm_eval"
    results = newest(list(root.glob("**/results_*.json")))
    if results is None:
        return None
    samples = {}
    for f in root.glob("**/samples_*.jsonl"):
        task = re.match(r"samples_(.+)_[^_]+$", f.stem).group(1)
        if task not in samples or f.name > samples[task].name:
            samples[task] = f
    return json.loads(results.read_text()), samples


def scores(results):
    """{(task, metric): (value, stderr, n)} from an lm-eval results json."""
    out = {}
    for task, r in results["results"].items():
        n = results.get("n-samples", {}).get(task, {}).get("effective")
        for key, value in r.items():
            if "," not in key or "_stderr," in key or not isinstance(value, float):
                continue
            metric, flt = key.split(",", 1)
            se = r.get("%s_stderr,%s" % (metric, flt))
            se = se if isinstance(se, float) else None
            out[(task, "%s (%s)" % (metric, flt))] = (value, se, n)
    return out


def per_question(samples_path):
    """{metric label: {doc_id: score}} from an lm-eval samples file."""
    out = {}
    # Split on "\n" only: str.splitlines() also breaks at U+2028 and friends, which
    # lm-eval writes unescaped inside the generated text, cutting a record in two.
    for line in samples_path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        s = json.loads(line)
        for metric in s.get("metrics", []):
            if isinstance(s.get(metric), (int, float)):
                label = "%s (%s)" % (metric, s.get("filter", "none"))
                out.setdefault(label, {})[s["doc_id"]] = float(s[metric])
    return out


def mcnemar_p(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt(v, se):
    if v is None:
        return "-"
    return "%.4f ± %.4f" % (v, se) if se is not None else "%.4f" % v


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    arms = arm_order(run_dir)
    ref, others = arms[0], arms[1:]
    tps = sorted(
        {int(re.match(r"\w+_tp(\d+)$", d.name).group(1)) for d in run_dir.glob("*_tp*")},
        reverse=True,
    )

    score_rows, paired_rows, missing = [], [], []
    for tp in tps:
        data = {}
        for tag in arms:
            arm = load_arm(run_dir / ("%s_tp%d" % (tag, tp)))
            if arm is None:
                missing.append("%s TP=%d" % (tag, tp))
            else:
                data[tag] = arm
        s = {tag: scores(results) for tag, (results, _) in data.items()}
        keys = sorted({k for v in s.values() for k in v})
        for task, metric in keys:
            cells = [str(tp), task, metric]
            n = next((v[(task, metric)][2] for v in s.values() if (task, metric) in v), None)
            cells.append(str(n) if n else "-")
            vals = {t: s.get(t, {}).get((task, metric), (None, None, None)) for t in arms}
            cells += [fmt(vals[t][0], vals[t][1]) for t in arms]
            for t in others:
                (rv, rse, _), (ov, ose, _) = vals[ref], vals[t]
                if rv is None or ov is None:
                    cells.append("-")
                elif rse is None or ose is None:
                    cells.append("%+.4f" % (ov - rv))
                else:
                    cells.append("%+.4f ± %.4f" % (ov - rv, math.hypot(rse, ose)))
            score_rows.append(cells)

        if ref not in data:
            continue
        ref_q = {task: per_question(p) for task, p in data[ref][1].items()}
        for t in others:
            if t not in data:
                continue
            for task, path in sorted(data[t][1].items()):
                if task not in ref_q:
                    continue
                arm_q = per_question(path)
                for metric in sorted(set(ref_q[task]) & set(arm_q)):
                    a, b = ref_q[task][metric], arm_q[metric]
                    docs = set(a) & set(b)
                    both = sum(1 for d in docs if a[d] and b[d])
                    neither = sum(1 for d in docs if not a[d] and not b[d])
                    ref_only = sum(1 for d in docs if a[d] and not b[d])
                    arm_only = sum(1 for d in docs if b[d] and not a[d])
                    paired_rows.append([
                        str(tp), task, metric, "%s vs %s" % (t, ref), str(len(docs)),
                        str(both), str(neither), str(ref_only), str(arm_only),
                        "%.3f" % mcnemar_p(ref_only, arm_only),
                    ])

    print("## Accuracy, reference %s\n" % ref)
    print("\n".join(table(
        ["TP", "task", "metric", "n"] + arms
        + ["%s − %s" % (t, ref) for t in others],
        score_rows,
    )))
    if paired_rows:
        print("\n## Paired, per question\n")
        print("\n".join(table(
            ["TP", "task", "metric", "arms", "n", "both right", "both wrong",
             "only %s right" % ref, "only the other right", "McNemar p"],
            paired_rows,
        )))
    if missing:
        print("\nNo lm-eval results: %s." % ", ".join(missing))


if __name__ == "__main__":
    main()
