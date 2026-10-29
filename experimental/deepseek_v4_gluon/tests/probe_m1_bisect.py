# SPDX-License-Identifier: MIT
"""CPU: the FFN launch's VGPRs / spills with the MoE off (seam only) and on."""
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.tests.probe_m1_ffn import ADDR, mock
from deepseek_v4_gluon.v41.ffn import kffn

for s in [int(a) for a in sys.argv[1:]] or [4, 6, 16]:
    for moe in (False, True):
        c = mock(s)
        c["MOE"] = moe
        sig = {n: "*u8" for n in ADDR} | {"rank": "i32", "epoch": "*u8", "tl": "i64",
                                          "health": "*u8"} | {k: "constexpr" for k in c}
        r = compile_probe(kffn, sig, c, enable_fp_fusion=False)
        print(f"S={s} moe={moe}: vgpr={r['vgpr']} spills={r['spills']}", flush=True)
