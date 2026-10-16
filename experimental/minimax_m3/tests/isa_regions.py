"""Instruction mix between consecutive timeline stamps of a traced Gluon M3 build.
python3 tests/isa_regions.py ISA.s KERNEL.py"""
import re
import sys
from collections import Counter

isa = open(sys.argv[1]).read().split("\n")
src = open(sys.argv[2]).read().split("\n")
marks = []
for i, l in enumerate(isa):
    if "s_memrealtime" in l:
        loc = [x for x in isa[:i] if ".loc" in x][-1]
        m = re.search(r"prims\.py:\d+:\d+ @\[ kernel\.py:(\d+)", loc)
        ln = int(m[1]) if m else 0
        st = re.search(r"stamp\(tr_, bid, (\d+)\)", src[ln - 1]) if ln else None
        marks.append((i, st[1] if st else "?"))
for (a, sa), (b, sb) in zip(marks, marks[1:]):
    c = Counter()
    for l in isa[a:b]:
        t = l.strip()
        if not t or t.startswith((".", ";")) or t.endswith(":"):
            continue
        c[t.split()[0]] += 1
    tot = sum(c.values())
    mem = {k: v for k, v in c.items() if k.startswith(("buffer_", "global_", "ds_", "s_barrier", "v_mfma", "s_waitcnt"))}
    print(f"{sa:>3} -> {sb:<3} {tot:6d} instr  " + " ".join(f"{k}:{v}" for k, v in sorted(mem.items(), key=lambda x: -x[1])[:8]))
