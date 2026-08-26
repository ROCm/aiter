#!/usr/bin/env python3
"""Mechanical pre-filter for the defect family that rule D9 originally missed.

The original D9 trigger was a name list (token_id, seq_start, batch_offset,
total_tokens). Three known int32-overflow defects used none of those names, so the rule
stayed silent. This scanner is now D9's structural pre-filter: an index-like value
multiplied by a stride-like value, on a line with no explicit 64-bit widening.

Output is a candidate list for the reviewer to judge -- not a verdict. Deliberately noisy
in the safe direction: it is a prompt to check, and a checked-and-dismissed candidate costs
one line of reasoning.

usage: scan_index_width.py <repo> <pr>        (reads the PR diff via gh)
       scan_index_width.py --diff <file> [--json]
"""

import argparse
import json
import re
import subprocess

INDEXY = re.compile(
    r"\b(\w*(?:idx|_id|pid|block|row|token|slot|page|seq|offset|off)\w*)\b",
    re.IGNORECASE,
)
STRIDEY = re.compile(
    r"\b(\w*stride\w*|\w*_pitch\w*|HiddenDim|hidden_dim|\w*_elems\b)\b", re.IGNORECASE
)
WIDENED = re.compile(
    r"(tl\.int64|gl\.int64|\.to\(\s*(?:tl|gl)\.int64\s*\)|int64_t|Int64|\.to\(torch\.int64\)|static_cast<\s*int64_t)",
    re.IGNORECASE,
)
SIGPARAM = re.compile(r"^\s*(\w*stride\w*|\w*_pitch\w*)\s*(,|:|\))", re.IGNORECASE)
KERNEL_EXT = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?")
    parser.add_argument("pr", nargs="?")
    parser.add_argument("--diff")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.diff:
        if args.repo or args.pr:
            parser.error("--diff cannot be combined with repo/pr")
    elif not args.repo or not args.pr:
        parser.error("provide either --diff FILE or REPO PR")
    return args


def get_diff(args):
    if args.diff:
        with open(args.diff, encoding="utf-8") as diff_file:
            return diff_file.read()
    result = subprocess.run(
        ["gh", "pr", "diff", args.pr, "--repo", args.repo],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def deduplicate(items, key):
    seen = set()
    result = []
    for item in items:
        item_key = key(item)
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


def scan(diff):
    cur, hits, sig_unwidened = None, [], []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
        if not line.startswith("+") or line.startswith("+++") or not cur:
            continue
        if not cur.endswith(KERNEL_EXT):
            continue
        code = line[1:]
        if WIDENED.search(code):
            continue
        # (a) a stride-ish kernel parameter declared with no width annotation
        m = SIGPARAM.match(code)
        if m and ":" not in code:
            sig_unwidened.append((cur, m.group(1), code.strip()[:90]))
        # (b) index * stride on one line, nothing widened
        if "*" in code:
            for expr in re.findall(r"([\w\.\[\]]+)\s*\*\s*([\w\.\[\]]+)", code):
                a, b = expr
                if (INDEXY.search(a) and STRIDEY.search(b)) or (
                    INDEXY.search(b) and STRIDEY.search(a)
                ):
                    hits.append((cur, f"{a} * {b}", code.strip()[:110]))
                    break
    return (
        deduplicate(hits, lambda item: (item[0], item[1])),
        deduplicate(sig_unwidened, lambda item: (item[0], item[1])),
    )


def main():
    args = parse_args()
    hits, sig_unwidened = scan(get_diff(args))
    if args.json:
        print(
            json.dumps(
                {
                    "index_stride_candidates": len(hits),
                    "untyped_stride_parameters": len(sig_unwidened),
                    "total_candidates": len(hits) + len(sig_unwidened),
                    "candidates": [
                        {"path": path, "expression": expression, "line": line}
                        for path, expression, line in hits
                    ],
                    "parameters": [
                        {"path": path, "name": name, "line": line}
                        for path, name, line in sig_unwidened
                    ],
                }
            )
        )
        return

    print(f"== index x stride with no 64-bit widening on the line: {len(hits)} ==")
    for f, e, c in hits:
        print(f"  {f}\n      {e:38} | {c}")
    print(
        f"\n== stride-like kernel params declared without a width annotation: {len(sig_unwidened)} =="
    )
    for f, p, c in sig_unwidened:
        print(f"  {f}: {p}")


if __name__ == "__main__":
    main()
