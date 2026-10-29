# SPDX-License-Identifier: MIT
"""CPU: compile the shared-expert stage (patched Triton image)."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512, deadline
from deepseek_v4_gluon.v41 import moe_shared as ms


@gluon.jit
def k_shared(sgu, sgu_s, scratch, health, rank, S: gl.constexpr):
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    sb = scratch
    xa = gl.allocate_shared_memory(gl.uint8, [4, 16, 8, 128], ms.SH4)
    xb = gl.allocate_shared_memory(gl.uint8, [16, 8, 128], ms.SH3)
    ca = gl.allocate_shared_memory(gl.uint64, [512], ms.SH1)
    cb = gl.allocate_shared_memory(gl.uint64, [128], ms.SH1)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    pair = gl.allocate_shared_memory(gl.float32, [512], SMEM_FLAT)
    tab = gl.allocate_shared_memory(gl.int32, [1024], SMEM_FLAT)
    t0 = (bid // 36) * 16
    n = gl.minimum(16, S - t0)
    task = bid % 36
    if t0 < S:
        ms.load_x_tile(sb, sb + (1 << 20), sb + (2 << 20), t0, n, xa, xb, ca, cb, tab, tag, tid,
                       dl, health, rank, 5120, False, 0, 6)
        ms.stage_shared(task, t0, n, sgu, sgu_s, sb + (3 << 20), sb + (4 << 20), sb + (5 << 20),
                        xa, xb, ca, cb, red, pair, tag, tid, dl, health, rank, 576, 5120)


for s in (1, 16, 48):
    r = compile_probe(k_shared, {"sgu": "*u8", "sgu_s": "*u8", "scratch": "*u8", "health": "*u8",
                                 "rank": "i32", "S": "constexpr"}, {"S": s}, enable_fp_fusion=False)
    print(f"S={s}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"scaled_mfma={r['asm'].count('v_mfma_scale_f32_16x16x128_f8f6f4')}", flush=True)
