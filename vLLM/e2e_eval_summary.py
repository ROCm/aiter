#!/usr/bin/env python3
"""Markdown summary of an `e2e.sh eval` run directory: every arm against the first.

    e2e_eval_summary.py <run dir>

The arms and their order come from <run dir>/arms.txt (the first is the
reference). Each arm's lm-eval output is under
<tag>_tp<N>/traces/lm_eval/<model>/{results_*.json,samples_<task>_*.jsonl},
one results file per lm-eval run (e2e.sh runs one for the short tasks and one
per long-context length); they are merged, a newer one winning a clash. Prints
two tables:
  - per (TP, task, metric) the score of each arm with its standard error, and
    the difference of every other arm from the first, with the standard error
    of the difference (the two runs taken as independent);
  - per (TP, task, metric) and arm, the paired comparison with the first arm on
    the same prompts: on how many each one scored higher, the two-sided exact
    sign-test p-value of that split (McNemar's test when scores are right or
    wrong; RULER's are fractions), the mean paired difference with its standard
    error, and how many generations were identical to the reference's
    (with the median character at which the others first differ). Greedy
    decoding makes the pairing tight, so this is the more sensitive test.
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


def load_arm(arm_dir):
    """(merged results, {task: samples paths, oldest first}) of one arm, or None.

    The results' "results" and "n-samples" are merged per task over all
    results_*.json, oldest first, so a newer run's metric wins. RULER reports
    -1 for the lengths a run did not have; those are not results and are dropped
    with their stderr, so that they do not overwrite another run's.
    """
    root = arm_dir / "traces" / "lm_eval"
    files = sorted(root.glob("**/results_*.json"), key=lambda p: p.name)
    if not files:
        return None
    merged = {"results": {}, "n-samples": {}}
    for f in files:
        r = json.loads(f.read_text())
        for task, v in r.get("results", {}).items():
            dead = {k for k, x in v.items() if "," in k and x == -1}
            dead |= {"%s_stderr,%s" % tuple(k.split(",", 1)) for k in dead}
            merged["results"].setdefault(task, {}).update(
                (k, x) for k, x in v.items() if k not in dead)
        for task, v in r.get("n-samples", {}).items():
            merged["n-samples"].setdefault(task, {}).update(v)
    samples = {}
    for f in sorted(root.glob("**/samples_*.jsonl"), key=lambda p: p.name):
        task = re.match(r"samples_(.+)_[^_]+$", f.stem).group(1)
        samples.setdefault(task, []).append(f)
    return merged, samples


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


def per_question(samples_paths):
    """({metric label: {prompt: score}}, {prompt: generation}) from samples files.

    Prompts are keyed by lm-eval's prompt hash, so the pairing holds across
    files and only pairs questions the arms were really asked alike.
    """
    out, gens = {}, {}
    for path in samples_paths:
        # Split on "\n" only: str.splitlines() also breaks at U+2028 and friends, which
        # lm-eval writes unescaped inside the generated text, cutting a record in two.
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            s = json.loads(line)
            key = s.get("prompt_hash", s["doc_id"])
            # Generations: [[text]] per question; anything else (log-likelihoods)
            # compared as its JSON.
            r = s.get("resps")
            gens[key] = (r[0][0] if r and r[0] and isinstance(r[0][0], str)
                         else json.dumps(r))
            for metric in s.get("metrics", []):
                # -1: a RULER length this question was not of.
                if isinstance(s.get(metric), (int, float)) and s[metric] != -1:
                    label = "%s (%s)" % (metric, s.get("filter", "none"))
                    out.setdefault(label, {})[key] = float(s[metric])
    return out, gens


def first_difference(a, b):
    """Index of the first character where two generations differ, or None."""
    if a == b:
        return None
    a, b = a or "", b or ""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def mcnemar_p(b, c):
    """Two-sided exact sign-test p-value for b and c discordant pairs (McNemar's
    test when the scores are right or wrong)."""
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
            for task, paths in sorted(data[t][1].items()):
                if task not in ref_q:
                    continue
                (ref_s, ref_g), (arm_s, arm_g) = ref_q[task], per_question(paths)
                for metric in sorted(set(ref_s) & set(arm_s)):
                    a, b = ref_s[metric], arm_s[metric]
                    docs = sorted(set(a) & set(b))
                    if not docs:
                        continue
                    ref_higher = sum(1 for d in docs if a[d] > b[d])
                    arm_higher = sum(1 for d in docs if b[d] > a[d])
                    diffs = [b[d] - a[d] for d in docs]
                    mean = sum(diffs) / len(docs)
                    se = (math.sqrt(sum((x - mean) ** 2 for x in diffs)
                                    / (len(docs) - 1) / len(docs))
                          if len(docs) > 1 else 0.0)
                    firsts = [first_difference(ref_g.get(d), arm_g.get(d)) for d in docs]
                    differ = sorted(i for i in firsts if i is not None)
                    paired_rows.append([
                        str(tp), task, metric, "%s vs %s" % (t, ref), str(len(docs)),
                        str(len(docs) - ref_higher - arm_higher), str(ref_higher),
                        str(arm_higher), "%.3f" % mcnemar_p(ref_higher, arm_higher),
                        "%+.4f ± %.4f" % (mean, se),
                        "%d/%d" % (len(docs) - len(differ), len(docs)),
                        str(differ[len(differ) // 2]) if differ else "-",
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
            ["TP", "task", "metric", "arms", "n", "same score",
             "%s higher" % ref, "other higher", "sign test p", "mean diff",
             "identical outputs", "median first diff (char)"],
            paired_rows,
        )))
    if missing:
        print("\nNo lm-eval results: %s." % ", ".join(missing))


if __name__ == "__main__":
    main()
