# SPDX-License-Identifier: MIT
"""CPU: the FFN launch's VGPRs / spills, its scalarized loads (byte / short
global loads by source line) and its operand shuffles (ds_bpermute, LDS
traffic) per width."""
import collections
import re
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.tests.probe_m1_ffn import ADDR, mock
from deepseek_v4_gluon.v41.ffn import kffn

for s in [int(a) for a in sys.argv[1:]] or [6, 16]:
    c = mock(s)
    sig = {n: "*u8" for n in ADDR} | {"rank": "i32", "epoch": "*u8", "tl": "i64",
                                      "health": "*u8"} | {k: "constexpr" for k in c}
    r = compile_probe(kffn, sig, c, enable_fp_fusion=False)
    asm = r["asm"]
    files = dict(re.findall(r'\.file\s+(\d+)\s+"[^"]*"\s+"([^"]+)"', asm))
    loc, small, perm = None, collections.Counter(), collections.Counter()
    for line in asm.splitlines():
        m = re.match(r"\s*\.loc\s+(\d+)\s+(\d+)", line)
        if m:
            loc = (files.get(m.group(1), "").split("/")[-1], int(m.group(2)))
            continue
        if not loc:
            continue
        if re.match(r"\s+global_load_(?:ubyte|sbyte|ushort|sshort|short_d16\w*|ubyte_d16\w*)", line):
            small[loc] += 1
        if re.match(r"\s+ds_bpermute", line):
            perm[loc] += 1
    n = lambda p: len(re.findall(p, asm))  # noqa: E731
    print(f"S={s}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"dx4={n(r'global_load_dwordx4')} ds_write={n(r'ds_write')} ds_read={n(r'ds_read')}\n"
          f"  small loads={sum(small.values())} {small.most_common(6)}\n"
          f"  bpermute={sum(perm.values())} {perm.most_common(6)}", flush=True)
