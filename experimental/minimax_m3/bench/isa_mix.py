"""Instruction mix of region-only builds, net of the empty build:
python3 bench/isa_mix.py REGION.s EMPTY.s [REGION2.s EMPTY2.s ...] (pairs; one column each)."""
import re
import sys
from collections import Counter

CATS = [
    ("total", lambda o: True),
    ("valu", lambda o: o.startswith("v_") and not o.startswith(("v_mfma", "v_readfirstlane", "v_readlane", "v_writelane"))),
    ("  v_dot2", lambda o: o.startswith("v_dot2")),
    ("  v_cvt/pk", lambda o: o.startswith(("v_cvt", "v_pk_", "v_perm", "v_bfe", "v_lshl_or", "v_and_or", "v_alignbit"))),
    ("  v_fma/mul/add f32", lambda o: re.match(r"v_(fma|fmac|mul|add|sub)_f32", o) is not None),
    ("  v_cndmask", lambda o: o.startswith("v_cndmask")),
    ("  dpp/permlane", lambda o: "dpp" in o or o.startswith(("v_permlane", "ds_swizzle", "ds_bpermute", "ds_permute"))),
    ("v_readfirstlane", lambda o: o.startswith(("v_readfirstlane", "v_readlane", "v_writelane"))),
    ("salu", lambda o: o.startswith("s_") and not o.startswith(("s_waitcnt", "s_barrier", "s_load", "s_buffer", "s_nop", "s_cbranch", "s_branch"))),
    ("s_load", lambda o: o.startswith(("s_load", "s_buffer_load"))),
    ("vmem load", lambda o: re.match(r"(buffer|global)_load", o) is not None),
    ("vmem store", lambda o: re.match(r"(buffer|global)_store", o) is not None),
    ("vmem atomic", lambda o: re.match(r"(buffer|global)_atomic", o) is not None),
    ("ds_read", lambda o: o.startswith("ds_read")),
    ("ds_write", lambda o: o.startswith("ds_write")),
    ("s_barrier", lambda o: o == "s_barrier"),
    ("s_waitcnt", lambda o: o.startswith("s_waitcnt")),
    ("s_nop", lambda o: o.startswith("s_nop")),
    ("branch", lambda o: o.startswith(("s_cbranch", "s_branch"))),
    ("mfma", lambda o: o.startswith("v_mfma")),
    ("scratch", lambda o: o.startswith("scratch_")),
]


def mix(path):
    c = Counter()
    meta = {}
    for l in open(path):
        t = l.strip()
        m = re.match(r"\.(vgpr_count|sgpr_count|agpr_count):\s+(\d+)", t) or re.match(r";\s*(NumVgprs|NumSgprs|ScratchSize):\s*(\d+)", t)
        if m:
            meta[m[1]] = int(m[2])
        if not t or t.startswith((".", ";", "//")) or t.endswith(":") or t.startswith("_"):
            continue
        op = t.split()[0]
        if not re.match(r"^[sv]_|^(buffer|global|ds|scratch|flat)_", op):
            continue
        c[op] += 1
    return c, meta


cols = []
for reg, emp in zip(sys.argv[1::2], sys.argv[2::2]):
    (cr, mr), (ce, _) = mix(reg), mix(emp)
    net = Counter({k: cr[k] - ce[k] for k in set(cr) | set(ce)})
    cols.append((reg.split("/")[-1][:28], net, mr))
print(f"{'':22s}" + "".join(f"{n:>30s}" for n, _, _ in cols))
for name, f in CATS:
    print(f"{name:22s}" + "".join(f"{sum(v for k, v in net.items() if f(k)):30d}" for _, net, _ in cols))
for key in ("NumVgprs", "NumSgprs", "ScratchSize", "vgpr_count", "sgpr_count"):
    if any(key in m for _, _, m in cols):
        print(f"{key:22s}" + "".join(f"{m.get(key, -1):30d}" for _, _, m in cols))
