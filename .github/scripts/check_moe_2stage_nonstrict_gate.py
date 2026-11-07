#!/usr/bin/env python3
"""Gate the MoE 2-stage full-matrix CI run against a known allowlist of
non-strict accuracy failures, instead of a bare count cap.
"""

import argparse
import csv
import re
import sys

FIELDS = [
    "token",
    "qType",
    "actType",
    "AQDType",
    "WQDType",
    "activation_bucket",
    "failure_mode",
]

# The fused_moe dispatcher's actual intermediate-activation dtype switches
# bf16->fp4 at larger token counts. This is NOT present in test_fmoe()'s
# logged call signature (its AQDType parameter stays constant across the
# whole sweep), so it can't be regex-extracted like the other fields.
# Verified manually against the known baseline log (main run 41552231943,
# job 125695520646): bf16 for token <= 128, fp4 once the dispatcher
# switches at token >= 256.
_BF16_ACTIVATION_TOKENS = {"1", "3", "5", "16", "32", "64", "128"}
_FP4_ACTIVATION_TOKENS = {"256", "1024", "4096", "8192", "163840"}


def activation_bucket_for_token(token):
    if token in _BF16_ACTIVATION_TOKENS:
        return "bf16"
    if token in _FP4_ACTIVATION_TOKENS:
        return "fp4"
    return "?"


def extract_failures(log_path):
    with open(log_path, errors="replace") as f:
        lines = f.readlines()

    fail_idxs = [
        i for i, l in enumerate(lines) if "accuracy check failed (non-strict)" in l
    ]
    failures = []
    for idx in fail_idxs:
        start = None
        for j in range(idx, max(-1, idx - 200), -1):
            if "calling test_fmoe(" in lines[j]:
                start = j
                break
        if start is None:
            failures.append(
                {
                    "_line": idx + 1,
                    "_raw": lines[idx].strip(),
                    "_unparsed": True,
                    "_reason": "no preceding test_fmoe( call found",
                }
            )
            continue
        block = []
        for j in range(start, idx):
            block.append(lines[j])
            if lines[j].rstrip().endswith(")"):
                break
        text = "".join(block)

        def grab(key, text=text):
            m = re.search(rf"{key}\s*=\s*([^\n,]+)", text)
            return m.group(1).strip().rstrip(",") if m else "?"

        raw = lines[idx].strip()
        failure_mode = "nan" if "output contains NaN" in raw else "numeric"
        sig = {
            k: grab(k) for k in FIELDS if k not in ("failure_mode", "activation_bucket")
        }
        sig["activation_bucket"] = activation_bucket_for_token(sig["token"])
        sig["failure_mode"] = failure_mode
        sig["_line"] = idx + 1
        sig["_raw"] = raw
        sig["_unparsed"] = False
        failures.append(sig)
    return failures


def load_allowlist(path):
    allowlist = set()
    with open(path, newline="") as f:
        data_lines = (
            line for line in f if not line.lstrip().startswith("#") and line.strip()
        )
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

    print(
        f"Observed {len(failures)} non-strict accuracy failures "
        f"({len(known)} known, {len(unexpected)} unexpected)."
    )

    if unexpected:
        print("\nUnexpected non-strict accuracy failures (not in allowlist):")
        for f in unexpected:
            if f["_unparsed"]:
                reason = f.get("_reason", "could not parse config")
                print(f"  line {f['_line']}: ({reason}) {f['_raw']}")
            else:
                sig_str = ", ".join(f"{k}={f[k]}" for k in FIELDS)
                print(f"  line {f['_line']}: {sig_str} -> {f['_raw']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
