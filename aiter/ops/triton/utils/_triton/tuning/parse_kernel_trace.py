"""Median kernel runtime per segment of a ``rocprofv3 --kernel-trace`` CSV.

A segment is everything between two ``split_dummy`` launches (see
``_worker.py``). Within a segment, the kernels whose names contain one of the
keywords are the op's kernels; each run's runtime is the sum over them.
"""

import argparse

import numpy as np
import pandas as pd


def segment_times(filename, keywords):
    """``(kernel names, median runtime in us)`` for each segment; the runtime is
    None when no kernel in the segment matches a keyword."""
    df_raw = pd.read_csv(filename)
    split_idx = (df_raw["Kernel_Name"] == "split_dummy").to_numpy().nonzero()[0]
    if len(split_idx) > 0:
        split_idx = np.insert(split_idx, 0, 0)
        segments = [
            df_raw.iloc[split_idx[i] : split_idx[i + 1]]
            for i in range(len(split_idx) - 1)
        ]
    else:
        segments = [df_raw]

    times = []
    for df in segments:
        names = [
            name
            for name in dict.fromkeys(df["Kernel_Name"])
            if any(key in name for key in keywords)
        ]
        if not names:
            times.append((names, None))
            continue
        durations = [
            (d["End_Timestamp"] - d["Start_Timestamp"]).to_numpy()
            for d in (df[df["Kernel_Name"] == name] for name in names)
        ]
        runs = min(len(d) for d in durations)
        runtime = sum(d[:runs] for d in durations) / 1e3
        times.append((names, float(runtime[np.argsort(runtime)[runs // 2]])))
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filename", type=str, help="rocprofv3 kernel-trace CSV")
    parser.add_argument(
        "-k", type=str, nargs="+", required=True, help="keywords of kernel names"
    )
    parser.add_argument("-m", type=float, help="Memory (GB)", default=0)
    parser.add_argument("-f", type=float, help="Operations (TFLOPS)", default=0)
    args = parser.parse_args()

    for names, runtime in segment_times(args.filename, args.k):
        print("Kernel detected:")
        for name in names:
            print(f"\t{name}")
        if runtime is None:
            print("no kernel matches the keywords")
            continue
        print(f"{runtime : .3f} (us)")
        if args.m > 0:
            print(f"{(args.m) : .6e} (GB)")
            print(f"{(args.m/runtime*1e6) : .2f} (GB/s)")
        if args.f > 0:
            print(f"{(args.f) : .6e} (TLOPS)")
            print(f"{(args.f/runtime*1e6) : .2f} (TLOPS/s)")


if __name__ == "__main__":
    main()
