"""Opcode counts per source line (.loc) of a Gluon ISA dump:
python3 bench/isa_lines.py ISA.s REGEX [N]  (ops matching REGEX; top N kernel.py / prims.py lines)"""
import re
import sys
from collections import Counter

pat = re.compile(sys.argv[2])
c = Counter()
files = {}
cur = "?"
for l in open(sys.argv[1]):
    t = l.strip()
    if t.startswith(".file"):
        m = re.match(r'\.file\s+(\d+)\s+"[^"]*"\s+"([^"]+)"', t) or re.match(r'\.file\s+(\d+)\s+"([^"]+)"', t)
        if m:
            files[m[1]] = m[2].split("/")[-1]
    elif t.startswith(".loc"):
        p = t.split()
        cur = f"{files.get(p[1], p[1])}:{p[2]}"
    elif t and not t.startswith((".", ";")) and not t.endswith(":"):
        op = t.split()[0]
        if pat.search(op):
            c[cur] += 1
for k, v in c.most_common(int(sys.argv[3]) if len(sys.argv) > 3 else 25):
    print(f"{v:6d}  {k}")
