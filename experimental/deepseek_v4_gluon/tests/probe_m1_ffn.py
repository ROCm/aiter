# SPDX-License-Identifier: MIT
"""CPU: compile the whole V4.1 FFN launch (seam + MoE) for gfx950 on the patched
Triton, mock region offsets (the real ones: ``ffn.build_constexprs`` on a GPU box)."""
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.v41.ffn import kffn
from deepseek_v4_gluon.v41.plan import Route

ADDR = ["part", "res_in", "post_in", "comb_in", "pre_in", "hc_fn", "hc_scale", "hc_base", "ffn_w",
        "res_out", "post_out", "comb_out", "pre_out", "gate_w", "bias", "w13", "w13_s", "w2",
        "w2_s", "sgu", "sgu_s", "sw2", "sw2_s", "out", "scratch", "sym", "peers"]


def mock(s, tp=4):
    r = Route(s, tp)
    g, per = r.router_groups
    regions = ["LIN", "PMIX", "NORMED", "XRDY", "X8R", "X8RS", "X8", "X8S", "X8RDY", "LOGIT",
               "ROUTE", "SMID", "SMQ", "SMC", "MID", "MIDS", "UGF"]
    c = {k: (i + 1) << 20 for i, k in enumerate(regions)}
    c.update(S=s, TP=tp, HIDDEN=5120, HALF=1 << 23, FFN_OFF=tp * s * 2560 * 8, EPOCH_COUNTERS=320,
             EXPERTS=384, TOPK=6, SCALE=1.5, ROUTER0=r.router0, ROUTE0=r.route0, SHARED0=r.shared0,
             UG0=r.ug0, ROUTER_TASKS=r.router_tasks, GROUPS=g, PER=per, MAX_U=r.max_u,
             UG_PARTS=r.ug_parts, FIRST0=r.first0, ORDER0=r.order0, EXPERT_SLOT0=r.expert_slot0,
             MASK0=r.mask0, QUEUE0=r.queue0, TILE_FIRST0=r.tile_first0, TILE_SLOT0=r.tile_slot0,
             MULTI=r.multi, SLOT_TILES=r.slot_tiles, TABW=1 << r.tab_words.bit_length(),
             SH_INTER=r.sh_inter, INTER=r.inter, INTER_REAL=r.inter_real,
             SCALE_COLS=r.down_scale_cols, SH_UNITS=r.shared_units, MOE=True)
    return c


def main():
    ok = True
    for s in [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16]:
        c = mock(s)
        sig = {n: "*u8" for n in ADDR} | {"rank": "i32", "epoch": "*u8", "tl": "i64", "health": "*u8"}
        sig |= {k: "constexpr" for k in c}
        r = compile_probe(kffn, sig, c, enable_fp_fusion=False)
        asm = r["asm"]
        print(f"S={s}: vgpr={r['vgpr']} sgpr={r['sgpr']} spills={r['spills']} lds={r['lds']} "
              f"scaled_mfma={asm.count('v_mfma_scale_f32_16x16x128_f8f6f4')} "
              f"bf16_mfma={asm.count('v_mfma_f32_16x16x32_bf16')} f32_mfma={asm.count('v_mfma_f32_16x16x4')}",
              flush=True)
        ok &= r["spills"] == 0 and r["lds"] <= 160 * 1024
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
