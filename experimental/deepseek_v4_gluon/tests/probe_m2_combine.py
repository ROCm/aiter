# SPDX-License-Identifier: MIT
"""CPU: compile K2's combine stage (patched Triton image)."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import T512, deadline
from deepseek_v4_gluon.v41 import back as B


@gluon.jit
def k_comb(klen, pos, sink, cos_sin, scratch, health, rank, S: gl.constexpr, HEADS: gl.constexpr):
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    for unit in range(bid, S * (HEADS // 8), 256):
        B.stage_combine(unit, klen, pos, sink, cos_sin, scratch, scratch + (1 << 24),
                        scratch + (2 << 24), scratch + (6 << 24), scratch + (3 << 24),
                        scratch + (4 << 24), scratch + (5 << 24), tag, tid, dl, health, rank, S,
                        HEADS, False)


SIG = {n: "*u8" for n in ("klen", "pos", "sink", "cos_sin", "scratch", "health")}
SIG |= {"rank": "i32", "S": "constexpr", "HEADS": "constexpr"}
for heads, s in [(16, 1), (16, 48), (32, 4), (32, 48)]:
    r = compile_probe(k_comb, SIG, {"S": s, "HEADS": heads}, enable_fp_fusion=False)
    print(f"heads={heads} S={s}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"bperm={r['asm'].count('ds_bpermute')} fma={r['asm'].count('v_fma_f32')}", flush=True)
