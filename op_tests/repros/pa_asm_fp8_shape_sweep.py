#!/usr/bin/env python3
# Sweep shapes to find which trigger the fp8 ASM PA OOB.
# Each (bs, ctx, qlen) is tested in a fresh forked Python process.

import argparse
import os
import subprocess
import sys


def run_one(bs, ctx, qlen, n_repeat=5, num_blocks=8192):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "pa_asm_fp8_repeat_call.py"),
        "--bs", str(bs),
        "--ctx", str(ctx),
        "--qlen", str(qlen),
        "--n-repeat", str(n_repeat),
        "--num-blocks", str(num_blocks),
    ]
    env = dict(os.environ)
    env["AMD_SERIALIZE_KERNEL"] = "3"
    env["HIP_LAUNCH_BLOCKING"] = "1"
    try:
        r = subprocess.run(cmd, env=env, timeout=60,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    out = r.stdout.decode(errors="ignore")
    if "ALL OK" in out:
        return "OK", out
    if "CRASH" in out or "HIP error" in out or "illegal memory" in out:
        # find first crash iter from "CRASH at iter=N"
        import re
        m = re.search(r"CRASH at iter=(\d+)", out)
        crash_iter = int(m.group(1)) if m else -1
        return f"CRASH@{crash_iter}", out
    return "UNKNOWN", out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-repeat", type=int, default=5)
    args = ap.parse_args()

    print(f"# fp8 ASM PA shape sweep (each cell = {args.n_repeat} repeats "
          f"of same call, fresh process)")
    print(f"# OK = no crash. CRASH@k = launch error surfaced at call k "
          f"(0-indexed; means call k-1 corrupted device).")
    print()

    qlens = [1, 2, 3, 4]
    ctx_lens = [128, 512, 1024, 2048, 4096, 6724, 8192, 12288, 16384]
    batch_sizes = [16, 32, 64, 96, 128]

    for qlen in qlens:
        print(f"## qlen={qlen}")
        header = "ctx \\ bs |" + "".join(f"{b:>10} |" for b in batch_sizes)
        print(header)
        print("-" * len(header))
        for ctx in ctx_lens:
            row = f"{ctx:>8} |"
            for bs in batch_sizes:
                tag, _ = run_one(bs, ctx, qlen, n_repeat=args.n_repeat)
                row += f"{tag:>10} |"
            print(row, flush=True)
        print()


if __name__ == "__main__":
    sys.exit(main())
