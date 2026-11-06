#!/usr/bin/env python3
"""Gate the MoE 2-stage full-matrix CI run against a known allowlist of
non-strict accuracy failures, instead of a bare count cap.

A count cap (e.g. "fail only if failures > 12") cannot tell a known
pre-existing failure apart from a brand-new one that happens to keep the
total under the cap -- a new failure could silently displace a known one.
This script instead extracts the (token, qType, actType, AQDType, WQDType)
signature preceding each "accuracy check failed (non-strict)" line and
requires every observed signature to appear in the allowlist. Any
unexpected signature (new config, or a known config with a changed
error magnitude bucket) fails the job.
"""
import argparse
import csv
import re
import sys


FIELDS = ["token", "qType", "actType", "AQDType", "WQDType", "failure_mode"]


def extract_failures(log_path):
    with open(log_path, errors="replace") as f:
        lines = f.readlines()

    fail_idxs = [i for i, l in enumerate(lines) if "accuracy check failed (non-strict)" in l]
    failures = []
    for idx in fail_idxs:
        start = None
        for j in range(idx, max(-1, idx - 200), -1):
            if "calling test_fmoe(" in lines[j]:
                start = j
                break
        if start is None:
            failures.append({"_line": idx + 1, "_raw": lines[idx].strip(), "_unparsed": True})
            continue
        block = []
        for j in range(start, idx):
            block.append(lines[j])
            if lines[j].rstrip().endswith(")"):
                break
        text = "".join(block)

        def grab(key):
            m = re.search(rf"{key}\s*=\s*([^\n,]+)", text)
            return m.group(1).strip().rstrip(",") if m else "?"

        raw = lines[idx].strip()
        failure_mode = "nan" if "output contains NaN" in raw else "numeric"
        sig = {k: grab(k) for k in FIELDS if k != "failure_mode"}
        sig["failure_mode"] = failure_mode
        sig["_line"] = idx + 1
        sig["_raw"] = raw
        sig["_unparsed"] = False
        failures.append(sig)
    return failures


def load_allowlist(path):
    allowlist = set()
    with open(path, newline="") as f:
        data_lines = (line for line in f if not line.lstrip().startswith("#") and line.strip())
        for row in csv.DictReader(data_lines):
            allowlist.add(tuple(row[k].strip() for k in FIELDS))
    return allowlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--allowlist", required=True)
    args = ap.parse_args()

    failures = extract_failures(args.log)
    allowlist = load_allowlist(args.allowlist)

    unexpected = []
    known = []
    for f in failures:
        if f["_unparsed"]:
            unexpected.append(f)
            continue
        sig = tuple(f[k] for k in FIELDS)
        if sig in allowlist:
            known.append(f)
        else:
            unexpected.append(f)

    print(f"Observed {len(failures)} non-strict accuracy failures "
          f"({len(known)} known, {len(unexpected)} unexpected).")

    if unexpected:
        print("\nUnexpected non-strict accuracy failures (not in allowlist):")
        for f in unexpected:
            if f["_unparsed"]:
                print(f"  line {f['_line']}: (could not parse config) {f['_raw']}")
            else:
                sig_str = ", ".join(f"{k}={f[k]}" for k in FIELDS)
                print(f"  line {f['_line']}: {sig_str} -> {f['_raw']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
