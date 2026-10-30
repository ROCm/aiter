"""Kernel duration and inter-kernel gap (next start - this end) from a rocprofv3 kernel trace CSV."""
import csv
import glob
import statistics
import sys

f = sorted(glob.glob(sys.argv[1] + "/**/*kernel_trace.csv", recursive=True))[-1]
rows = list(csv.DictReader(open(f)))
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
k = [r for r in rows if "add" in r["Kernel_Name"].lower() or "elementwise" in r["Kernel_Name"].lower()]
k = k[-1000:]
dur = [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in k]
gap = [int(b["Start_Timestamp"]) - int(a["End_Timestamp"]) for a, b in zip(k, k[1:])]
print(f"{len(k)} kernels: duration p50 {statistics.median(dur) / 1e3:.3f} us, gap p50 {statistics.median(gap) / 1e3:.3f} us"
      f" (p10 {sorted(gap)[len(gap) // 10] / 1e3:.3f}, p90 {sorted(gap)[9 * len(gap) // 10] / 1e3:.3f})")
