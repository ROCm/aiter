# SPDX-License-Identifier: MIT
"""CPU: compile the ug stage (patched Triton image)."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512
from deepseek_v4_gluon.v41 import moe_shared as ms
from deepseek_v4_gluon.v41 import moe_ug as mu
from deepseek_v4_gluon.v41.plan import Route


@gluon.jit
def k_ug(w13, w13_s, scratch, S: gl.constexpr, UG_PARTS: gl.constexpr, FIRST0: gl.constexpr,
         ORDER0: gl.constexpr, TABW: gl.constexpr):
    tid = gl.arange(0, 512, layout=T512)
    lane = tid % 64
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    sb = scratch
    xa = gl.allocate_shared_memory(gl.uint8, [4, 16, 8, 128], ms.SH4)
    xb = gl.allocate_shared_memory(gl.uint8, [16, 8, 128], ms.SH3)
    ca = gl.allocate_shared_memory(gl.uint64, [512], ms.SH1)
    cb = gl.allocate_shared_memory(gl.uint64, [128], ms.SH1)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    tab = gl.allocate_shared_memory(gl.int32, [TABW], SMEM_FLAT)
    for task in range(bid, 64 * UG_PARTS, 256):
        e, first, count = mu.slot_unit(tab, task // UG_PARTS, tid, S, 6, FIRST0)
        mu.stage_ug(task, e, first, count, tab, w13, w13_s, sb, sb + (1 << 20), sb + (2 << 20), xa,
                    xb, ca, cb, red, tag, tid, lane, 6, UG_PARTS, ORDER0, 1280, 640, 5120, False)


for s in (1, 16):
    r = Route(s, 4)
    res = compile_probe(k_ug, {"w13": "*u8", "w13_s": "*u8", "scratch": "*u8", "S": "constexpr",
                               "UG_PARTS": "constexpr", "FIRST0": "constexpr", "ORDER0": "constexpr",
                               "TABW": "constexpr"},
                        {"S": s, "UG_PARTS": r.ug_parts, "FIRST0": r.first0, "ORDER0": r.order0,
                         "TABW": 1 << r.tab_words.bit_length()}, enable_fp_fusion=False)
    print(f"S={s}: vgpr={res['vgpr']} spills={res['spills']} lds={res['lds']} "
          f"scaled_mfma={res['asm'].count('v_mfma_scale_f32_16x16x128_f8f6f4')}", flush=True)
