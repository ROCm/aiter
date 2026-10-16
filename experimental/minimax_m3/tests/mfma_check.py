"""Flag v_mfma_scale instructions whose destination overlaps a source other than an
identical src C (ISA files given as arguments)."""
import re
import sys


def regs(tok):
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", tok)
    if m:
        return set(range(int(m[1]), int(m[2]) + 1))
    m = re.fullmatch(r"v(\d+)", tok)
    return {int(m[1])} if m else set()


for f in sys.argv[1:]:
    bad = n = 0
    for i, line in enumerate(open(f)):
        if "v_mfma" not in line:
            continue
        n += 1
        ops = [o.strip() for o in line.split(None, 1)[1].split(",")]
        ops = [o.split()[0] for o in ops]
        d, a, b, c = ops[:4]
        rest = ops[4:6]
        dr = regs(d)
        srcs = regs(a) | regs(b) | set().union(*[regs(x) for x in rest])
        if dr & srcs or (regs(c) and regs(c) != dr and regs(c) & dr):
            bad += 1
            if bad <= 5:
                print(f"{f}:{i + 1}: {line.strip()}")
    print(f"{f}: {n} mfma, {bad} overlapping")
