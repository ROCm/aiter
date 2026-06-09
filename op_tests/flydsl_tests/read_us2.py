# Parse rocprofv3 --kernel-trace sqlite output and report per-kernel device time.
#
#   python read_us2.py <rocprof_outdir> <kernel_name_substr> [stat]
#
# stat (default "p10"): p10 | min | median | all
#
# IMPORTANT: use p10 (10th-percentile), NOT median, for autotuning kernels.
# An autotuning provider (e.g. the upstream Triton jdbba kernel) fires hundreds
# to thousands of trial-config dispatches plus warmup; the slow trials inflate
# the median by 30-40% and make a non-autotuning kernel look artificially
# faster. p10 approximates the steady-state best-config time that do_bench and a
# real deployment see, and matches a non-autotuning kernel's clean runs.
import sys
import sqlite3
import statistics
import glob
import os

db = glob.glob(os.path.join(sys.argv[1], "**", "*.db"), recursive=True)
sub = sys.argv[2]
stat = sys.argv[3] if len(sys.argv) > 3 else "p10"
if not db:
    print("nan")
    sys.exit(0)
c = sqlite3.connect(db[0])
tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
disp = [t for t in tabs if "kernel_dispatch" in t]
sym = [t for t in tabs if "kernel_symbol" in t]
if not disp or not sym:
    print("nan")
    sys.exit(0)
rows = c.execute(
    f"SELECT s.kernel_name, d.end-d.start FROM {disp[0]} d JOIN {sym[0]} s ON d.kernel_id=s.id"
).fetchall()
d = sorted(v for nm, v in rows if v and v > 0 and sub in (nm or ""))
if not d:
    print("nan")
    sys.exit(0)
if stat == "all":
    print(
        f"n={len(d)} min={d[0]/1000:.2f} p10={d[len(d)//10]/1000:.2f} "
        f"median={statistics.median(d)/1000:.2f} max={d[-1]/1000:.2f}"
    )
elif stat == "min":
    print(f"{d[0]/1000:.2f}")
elif stat == "median":
    print(f"{statistics.median(d)/1000:.2f}")
else:  # p10 (default, steady-state best config)
    print(f"{d[len(d)//10]/1000:.2f}")
