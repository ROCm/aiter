# SPDX-License-Identifier: MIT
"""CPU-only: compile the M0 unit kernels for gfx950 and check resources and
the hand-off ISA (cache bits, polls, clock, no spills)."""

import re
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.tests.m0_kernels import k_ring

SIG = {
    "scratch": "i64",
    "health": "i64",
    "epoch": "i64",
    "out": "*fp32",
    "rank": "i32",
    "SYSTEM": "constexpr",
}

ok = True
for system in (False, True):
    r = compile_probe(k_ring, SIG, {"SYSTEM": system})
    asm = r["asm"]
    bits = "sc0 sc1" if system else "sc1"
    checks = {
        "tagged store": re.search(rf"global_store_dwordx2 .*off {bits}", asm),
        "poll load": re.search(rf"global_load_dwordx2 .*off {bits}", asm),
        "mark store sc1": re.search(r"global_store_dword v\[\d+:\d+\], v\d+, off sc1", asm),
        "health load": re.search(r"global_load_dword .*off sc0 sc1", asm),
        "clock": "s_memrealtime" in asm,
        "bpermute": "ds_bpermute_b32" in asm,
        "drain": re.search(r"s_waitcnt vmcnt\(0\)", asm),
        "barrier": "s_barrier" in asm,
        "no spills": r["spills"] == 0,
    }
    print(f"SYSTEM={system}: vgpr={r['vgpr']} sgpr={r['sgpr']} spills={r['spills']} "
          f"lds={r['lds']}")  # fmt: skip
    for name, good in checks.items():
        print(f"  {'OK ' if good else 'BAD'} {name}")
        ok &= bool(good)
sys.exit(0 if ok else 1)
