#!/usr/bin/env python3
"""Mechanical pre-filter for the defect family that rule D9 names but never fired on.

D9's trigger is a NAME LIST (token_id, seq_start, batch_offset, total_tokens). The three
int32-overflow defects in this eval used none of those names, so the rule stayed silent in
both arms. This scans the diff structurally instead: an index-like value multiplied by a
stride-like value, on a line with no explicit 64-bit widening.

Output is a candidate list for the reviewer to judge -- not a verdict. Deliberately noisy
in the safe direction: it is a prompt to check, and a checked-and-dismissed candidate costs
one line of reasoning.

usage: scan_index_width.py <repo> <pr>        (reads the PR diff via gh)
       scan_index_width.py --diff <file>
"""
import re, subprocess, sys

INDEXY = re.compile(r"\b(\w*(?:idx|_id|pid|block|row|token|slot|page|seq|offset|off)\w*)\b", re.I)
STRIDEY = re.compile(r"\b(\w*stride\w*|\w*_pitch\w*|HiddenDim|hidden_dim|\w*_elems\b)\b", re.I)
WIDENED = re.compile(r"(tl\.int64|gl\.int64|\.to\(\s*(?:tl|gl)\.int64\s*\)|int64_t|Int64|\.to\(torch\.int64\)|static_cast<\s*int64_t)", re.I)
SIGPARAM = re.compile(r"^\s*(\w*stride\w*|\w*_pitch\w*)\s*(,|:|\))", re.I)
KERNEL_EXT = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp")

def get_diff(argv):
    if argv[0] == "--diff":
        return open(argv[1]).read()
    return subprocess.run(["gh", "pr", "diff", argv[1], "--repo", argv[0]],
                          capture_output=True, text=True).stdout

def main():
    diff = get_diff(sys.argv[1:])
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
                if (INDEXY.search(a) and STRIDEY.search(b)) or (INDEXY.search(b) and STRIDEY.search(a)):
                    hits.append((cur, f"{a} * {b}", code.strip()[:110]))
                    break
    print(f"== index x stride with no 64-bit widening on the line: {len(hits)} ==")
    seen = set()
    for f, e, c in hits:
        k = (f, e)
        if k in seen:
            continue
        seen.add(k)
        print(f"  {f}\n      {e:38} | {c}")
    print(f"\n== stride-like kernel params declared without a width annotation: {len(sig_unwidened)} ==")
    seen = set()
    for f, p, c in sig_unwidened:
        if (f, p) in seen:
            continue
        seen.add((f, p))
        print(f"  {f}: {p}")

if __name__ == "__main__":
    main()
