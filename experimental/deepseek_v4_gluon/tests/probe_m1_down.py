# SPDX-License-Identifier: MIT
"""CPU: compile the down / combine / reduce stages (patched Triton image)."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512, deadline
from deepseek_v4_gluon.v41 import moe_down as md
from deepseek_v4_gluon.v41 import moe_shared as ms
from deepseek_v4_gluon.v41.plan import Route


@gluon.jit
def k_down(w2, w2_s, sw2, sw2_s, scratch, sym, peers, out, health, rank,
           S: gl.constexpr, UG_PARTS: gl.constexpr, FIRST0: gl.constexpr, ORDER0: gl.constexpr,
           TABW: gl.constexpr, CW: gl.constexpr, SDW: gl.constexpr):
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    sb = scratch
    pp = peers.to(gl.pointer_type(gl.int64))
    peer = (gl.load(pp), gl.load(pp + 1), gl.load(pp + 2), gl.load(pp + 3))
    tab = gl.allocate_shared_memory(gl.int32, [TABW], SMEM_FLAT)
    contrib = gl.allocate_shared_memory(gl.float32, [CW], SMEM_FLAT)
    sx32 = gl.allocate_shared_memory(gl.uint32, [4096], ms.SH1)
    sx8 = sx32._reinterpret(gl.uint8, [8, 16, 128], ms.SH3)
    sc8 = gl.allocate_shared_memory(gl.uint8, [8, 16, 4], ms.SH3)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    pair = gl.allocate_shared_memory(gl.float32, [512], SMEM_FLAT)
    sdl = gl.allocate_shared_memory(gl.float32, [SDW], SMEM_FLAT)
    task0 = bid
    rk = bid % 128
    md.load_smid_tile(sb, sb + (1 << 20), 0, S, sx32, sx8, sc8, tag, tid, dl, health, rank, 576)
    r = tid % 16
    tl = gl.minimum(tid // 16, 15)
    v = ms.wave_sum(md.down_shared_task(task0, sw2, sw2_s, sx8, sc8, 576, 5120), red, r, tl)
    sdl.scatter(gl.where(tid < 16 * S, v, 0.0), gl.where(tid < 16 * S, tl * 16 + r, SDW - 1), 0)
    if rk < 64:
        md.stage_down(task0, 256 + rk, tab, w2, w2_s, sb + (2 << 20), sb + (3 << 20), sb + (4 << 20),
                      contrib, tag, tid, dl, health, rank, 2, S, 6, FIRST0, ORDER0, UG_PARTS, 640,
                      576, 5120, 24, False, 1, 0)
    else:
        md.stage_down(task0, task0, tab, w2, w2_s, sb + (2 << 20), sb + (3 << 20), sb + (4 << 20),
                      contrib, tag, tid, dl, health, rank, 1, S, 6, FIRST0, ORDER0, UG_PARTS, 640,
                      576, 5120, 24, False, 1, 0)
    md.down_combine(task0, 0, 0, S, tab, contrib, sdl, pair, peer, rank, tag, tid, S, 6, 4, 5120,
                    4 * S * 2560 * 8)
    for task in range(bid, 160, 256):
        md.stage_reduce(task, sym, out, tag, tid, dl, health, rank, S, 4, 5120, 4 * S * 2560 * 8)


def p2(n):
    return 1 << (n - 1).bit_length()


for s in (1, 16):
    r = Route(s, 4)
    c = {"S": s, "UG_PARTS": r.ug_parts, "FIRST0": r.first0, "ORDER0": r.order0,
         "TABW": p2(r.tab_words + 1), "CW": p2(2 * r.picks * 16 + 1), "SDW": p2(2 * s * 16 + 1)}
    sig = {n: "*u8" for n in ("w2", "w2_s", "sw2", "sw2_s", "scratch", "sym", "peers", "out",
                              "health")} | {"rank": "i32"} | {k: "constexpr" for k in c}
    res = compile_probe(k_down, sig, c, enable_fp_fusion=False)
    print(f"S={s}: vgpr={res['vgpr']} spills={res['spills']} lds={res['lds']} "
          f"scaled_mfma={res['asm'].count('v_mfma_scale_f32_16x16x128_f8f6f4')}", flush=True)
