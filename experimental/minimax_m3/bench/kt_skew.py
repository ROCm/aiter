"""Cross-rank skew of the m3_step launches from rocprofv3 kernel traces (bench/ktrace_wrap.sh): the
timestamps share one clock domain, so launch k's start / end on the 4 ranks can be compared.
Per launch: start spread (max - min over ranks), and each rank's start - min start; plus the
rank-median of (latest start - own start) = how long a rank's first cross-rank wait idles at most.
python bench/kt_skew.py bench/logs/ktrace/NAME [last N launches]"""
import csv, glob, statistics as st, sys

d = sys.argv[1]
nlast = int(sys.argv[2]) if len(sys.argv) > 2 else 600
per = []
for r in range(4):
    f = glob.glob(f"{d}/r{r}/**/*kernel_trace.csv", recursive=True)[0]
    ks = [(int(x["Start_Timestamp"]), int(x["End_Timestamp"])) for x in csv.DictReader(open(f))
          if "m3_step" in x["Kernel_Name"]]
    per.append(ks[-nlast:])
n = min(len(p) for p in per)
spread, late, endspread = [], [[] for _ in range(4)], []
for k in range(n):
    s = [per[r][k][0] for r in range(4)]
    e = [per[r][k][1] for r in range(4)]
    spread.append((max(s) - min(s)) / 1e3)
    endspread.append((max(e) - min(e)) / 1e3)
    for r in range(4):
        late[r].append((max(s) - s[r]) / 1e3)
print(f"{d}: {n} launches; start spread med {st.median(spread):.2f} us (p90 {sorted(spread)[int(.9 * n)]:.2f}), "
      f"end spread med {st.median(endspread):.2f} us; per rank median (latest start - own start): "
      + " ".join(f"{st.median(late[r]):.2f}" for r in range(4)))
