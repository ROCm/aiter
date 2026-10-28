"""The traced kernel's disassembly (ATT code.json) for instructions whose source chain passes
through FUNC (e.g. stage_norm): per (innermost kernel.py/prims.py line, function) instruction counts
and opcodes, in address order; --dump prints the instructions with their lines.
    python3 asm_region.py DISPATCH_DIR FUNC [--dump] [--hits]"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from att_cta import parse_chain
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("func")
ap.add_argument("--dump", action="store_true"); ap.add_argument("--hits", action="store_true")
a = ap.parse_args()
code = json.load(open(f"{a.d}/code.json"))["code"]
rows = []
for c in code:
    if not c[3]:
        continue
    ch = parse_chain(c[3])
    funcs = [x[2] for x in ch]
    if a.func not in funcs:
        continue
    inner = ch[0]
    rows.append((c[5], c[0], inner, c[6], c[7], c[8]))  # vaddr, isa, (file, line, func), hit, latency, stall
rows.sort()
print(f"{a.func}: {len(rows)} instructions")
if a.dump:
    for va, isa, (f, l, fn), hit, lat, stall in rows:
        print(f"  {va:6x}  {f}:{l:<5d} {fn:<16s} {isa:<60s}" + (f" hit {hit} lat {lat} stall {stall}" if a.hits else ""))
else:
    per = collections.OrderedDict()
    for va, isa, (f, l, fn), hit, lat, stall in rows:
        k = (f, l, fn)
        if k not in per:
            per[k] = [collections.Counter(), 0, 0]
        per[k][0][isa.split()[0]] += 1
        per[k][1] += lat; per[k][2] += stall
    for (f, l, fn), (c, lat, stall) in per.items():
        print(f"  {f}:{l:<5d} {fn:<16s} n={sum(c.values()):<4d} lat {lat:8d} stall {stall:8d}  " + " ".join(f"{o}:{v}" for o, v in c.most_common(5)))
