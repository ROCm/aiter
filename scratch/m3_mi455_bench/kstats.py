"""Print VGPR / spill / scratch / LDS of the AR kernels from llvm-readelf --notes output."""
import re
import sys

t = open(sys.argv[1]).read()
for b in re.split(r"\n  - \.(?=agpr_count|args)", t):
    m = re.search(r"\.name:\s+(\S+)", b)
    if not m or not re.search(sys.argv[2] if len(sys.argv) > 2 else "ar_ll|ar_rmsnorm_gfx1250", m.group(1)):
        continue

    def g(k):
        x = re.search(r"\." + k + r":\s+(\S+)", b)
        return x.group(1) if x else "?"

    print(f"{m.group(1)[:80]:80} vgpr={g('vgpr_count')} sgpr={g('sgpr_count')} "
          f"vspill={g('vgpr_spill_count')} scratch={g('private_segment_fixed_size')} lds={g('group_segment_fixed_size')}")
