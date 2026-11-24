# print the TP4 tuned-row override (MEGA_CFG syntax) serving <tokens>: rowcfg.py <model> <mode> <tokens> [key]
import sys
from aiter.ops.flydsl.kernels.mega_moe_tp_1250.engine import _tuned_rows, _tuned_key
SH = {"dsv41": (5120, (384, 576, 4)), "m31": (6144, (129, 768, 4)), "kimi31": (3584, (896, 768, 4))}
md, sh = SH[sys.argv[1]]; mode = sys.argv[2]; t = int(sys.argv[3])
k = int(sys.argv[4]) if len(sys.argv) > 4 else _tuned_key(t, md, sh, mode)
r = _tuned_rows()
print(",".join(filter(None, [r.get((md, *sh, k, ""), ""), r.get((md, *sh, k, mode), "")])))
