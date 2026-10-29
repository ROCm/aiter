# SPDX-License-Identifier: MIT
import sys
from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.tests.m1_front_kernels import k_front
from deepseek_v4_gluon.v41.plan import Route
from deepseek_v4_gluon.v41.seam import pow2

ok = True
for s in [int(a) for a in sys.argv[1:]] or [1, 16, 48]:
    r = Route(s, 4)
    g, per = r.router_groups
    c = dict(S=s, HIDDEN=5120, EXPERTS=384, TOPK=6, GROUPS=g, PER=per, ROUTER0=r.router0,
             ROUTE0=r.route0, ROUTER_TASKS=r.router_tasks, MAX_U=r.max_u, MULTI=r.multi,
             SLOT_TILES=r.slot_tiles, EXPERT_SLOT0=r.expert_slot0, ORDER0=r.order0,
             FIRST0=r.first0, MASK0=r.mask0, TILE_FIRST0=r.tile_first0,
             TILE_SLOT0=r.tile_slot0, TABW=1 << (r.tab_words).bit_length(),
             X8R=0, X8RS=1 << 20, X8=2 << 20, X8S=3 << 20, X8RDY=4 << 20, LOGIT=5 << 20,
             ROUTE=6 << 20, XRDY=7 << 20)
    sig = {n: "i64" for n in ("x", "gate_w", "bias", "scratch", "epoch", "health")}
    sig |= {"rank": "i32", "tabdump": "i64"} | {k: "constexpr" for k in c}
    res = compile_probe(k_front, sig, c, enable_fp_fusion=False)
    asm = res["asm"]
    print(f"S={s}: vgpr={res['vgpr']} sgpr={res['sgpr']} spills={res['spills']} "
          f"lds={res['lds']} mfma_bf16={asm.count('v_mfma_f32_16x16x32_bf16')} "
          f"cvt_fp8={asm.count('v_cvt_pk_fp8_f32')} tab_words={r.tab_words}", flush=True)
    ok &= res["spills"] == 0
sys.exit(0 if ok else 1)
