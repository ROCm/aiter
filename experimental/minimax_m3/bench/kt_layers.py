"""Per-layer kernel times from a rocprofv3 kernel trace of the hybrid bench (rank dir):
the last timed replays' sequences of [m3_step_kernel, MoE kernels...], averaged by position."""
import csv, sys, collections, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
ks = [(r["Kernel_Name"], int(r["Start_Timestamp"]), int(r["End_Timestamp"])) for r in rows]
# keep the tail (timed graph replays)
ks = ks[-int(sys.argv[2]) if len(sys.argv) > 2 else -4000:]
short = lambda n: ("m3_step" if "m3_step" in n else n.split("(")[0].split("<")[0][-60:])
dur = collections.defaultdict(list)
gap = collections.defaultdict(list)
for i, (n, s, e) in enumerate(ks):
    dur[short(n)].append((e - s) / 1e3)
    if i:
        gap[short(n)].append((s - ks[i - 1][2]) / 1e3)
for n in dur:
    print(f"{n:62s} n={len(dur[n]):5d} dur med {statistics.median(dur[n]):7.2f} us   gap before med {statistics.median(gap[n]) if gap[n] else 0:6.2f} us")
