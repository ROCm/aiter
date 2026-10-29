# SPDX-License-Identifier: MIT
"""CPU: compile K2's wo_a and wo_b stages (patched Triton image)."""
import re

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512, deadline
from deepseek_v4_gluon.v41 import back as B


@gluon.jit
def k_gemv(woa, woa_s, wob, wob_s, scratch, peers, health, rank, S: gl.constexpr,
           HEADS: gl.constexpr, TP: gl.constexpr):  # fmt: skip
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    pp = peers.to(gl.pointer_type(gl.int64))
    peer = ()
    for p in gl.static_range(TP):
        peer = peer + (gl.load(pp + p),)
    x_s = gl.allocate_shared_memory(gl.uint8, [65536], B.SH1)
    c_s = gl.allocate_shared_memory(gl.uint8, [2048], B.SH1)
    red = gl.allocate_shared_memory(gl.float32, [4096], SMEM_FLAT)
    sb = scratch
    if bid < (8 // TP) * 32:
        B.stage_woa(bid, woa, woa_s, sb, sb + (1 << 22), sb + (2 << 22), sb + (3 << 22),
                    sb + (4 << 22), x_s, c_s, red, tag, tid, dl, health, rank, S, HEADS, TP)
    task2 = bid + 256
    B.stage_wob(bid, gl.minimum(task2, 319), task2 < 320, wob, wob_s, sb + (2 << 22),
                sb + (3 << 22), sb + (5 << 22), peer, x_s, c_s, red, tag, tid, dl, health, rank,
                S, HEADS, TP, 5120)


SIG = {n: "*u8" for n in ("woa", "woa_s", "wob", "wob_s", "scratch", "peers", "health")}
SIG |= {"rank": "i32", "S": "constexpr", "HEADS": "constexpr", "TP": "constexpr"}
for tp, s in [(4, 1), (4, 48), (2, 16)]:
    r = compile_probe(k_gemv, SIG, {"S": s, "HEADS": 64 // tp, "TP": tp}, enable_fp_fusion=False)
    a = r["asm"]
    n = lambda p: len(re.findall(p, a))  # noqa: E731
    print(f"tp={tp} S={s}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"smfma={n('v_mfma_scale_f32_16x16x128')} bperm={n('ds_bpermute')} "
          f"ubyte={n('global_load_ubyte')} dx4={n('global_load_dwordx4')}", flush=True)
