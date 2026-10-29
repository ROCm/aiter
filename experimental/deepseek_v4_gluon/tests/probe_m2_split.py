# SPDX-License-Identifier: MIT
"""CPU: compile K2's split stage (patched Triton image) per (TP heads, S, ratio)."""
import re
import sys

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.common.sync import SMEM_FLAT, T512, deadline
from deepseek_v4_gluon.v41 import back as B


@gluon.constexpr_function
def p2(n):
    return 1 << (n - 1).bit_length()


@gluon.jit
def k_split(q, kt, klen, swa, swa_stride, swa_block, comp, comp_stride, comp_block, zrec,
            qk_scale, scratch, health, rank, S: gl.constexpr, HEADS: gl.constexpr,
            RATIO: gl.constexpr):  # fmt: skip
    tid = gl.arange(0, 512, layout=T512)
    bid = gl.program_id(0)
    tag = 2 + gl.zeros_like(tid)
    dl = deadline(tid)
    NG: gl.constexpr = B.head_groups(HEADS, S)
    CK: gl.constexpr = B.chunk_keys(HEADS, S)
    NCHUNK: gl.constexpr = B.n_chunks(HEADS, S)
    HT: gl.constexpr = HEADS // 16
    NKT: gl.constexpr = 8 // NG
    q_s = gl.allocate_shared_memory(gl.bfloat16, [16, HT, 16, 32], B.SH4)
    raw_n = gl.allocate_shared_memory(gl.uint8, [16, NG, CK, 32], B.SH4)
    raw_r = gl.allocate_shared_memory(gl.bfloat16, [2, NG, CK, 32], B.SH4)
    raw_s = gl.allocate_shared_memory(gl.uint8, [NG, CK, 32], B.SH3)
    p_s = gl.allocate_shared_memory(gl.bfloat16, [HT, 16, CK], B.SH3)
    st_s = gl.allocate_shared_memory(gl.float32, [p2(HEADS * (1 + NKT))], SMEM_FLAT)
    for unit in range(bid, S * NCHUNK, 256):
        B.stage_split(unit, q, kt, klen, swa, swa_stride, swa_block, comp, comp_stride,
                      comp_block, zrec, qk_scale, scratch, scratch + (1 << 24),
                      scratch + (2 << 24), scratch + (3 << 24), q_s, raw_n, raw_r, raw_s, p_s, st_s, tag, tid, dl,
                      health, rank, S, HEADS, RATIO, False)  # fmt: skip


SIG = {n: "*u8" for n in ("q", "kt", "klen", "swa", "comp", "zrec", "scratch", "health")}
SIG |= {"swa_stride": "i64", "comp_stride": "i64", "swa_block": "i32", "comp_block": "i32",
        "qk_scale": "i32", "rank": "i32", "S": "constexpr", "HEADS": "constexpr",
        "RATIO": "constexpr"}
for heads, s, ratio in [(16, 1, 1), (16, 48, 0), (32, 4, 1), (32, 48, 1)]:
    try:
        r = compile_probe(k_split, SIG, {"S": s, "HEADS": heads, "RATIO": ratio},
                          enable_fp_fusion=False)  # fmt: skip
        a = r["asm"]
        n = lambda p: len(re.findall(p, a))  # noqa: E731
        print(f"heads={heads} S={s} ratio={ratio}: vgpr={r['vgpr']} spills={r['spills']} "
              f"lds={r['lds']} mfma32={n('v_mfma_f32_16x16x32_bf16')} "
              f"mfma16={n('v_mfma_f32_16x16x16_?bf16')} bperm={n('ds_bpermute')} "
              f"ubyte={n('global_load_ubyte')} dx2={n('global_load_dwordx2')}", flush=True)
    except Exception as e:
        print(f"heads={heads} S={s}: FAIL {str(e).strip().splitlines()[-1][:400]}", flush=True)
        if "-v" in sys.argv:
            raise
