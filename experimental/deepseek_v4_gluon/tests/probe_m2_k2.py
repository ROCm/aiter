# SPDX-License-Identifier: MIT
"""CPU: compile the whole V4.1 K2 launch (attention back + FFN half) for gfx950
on the patched Triton, mock region offsets (real ones: ``k2.build_constexprs``)."""
import re
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.tests.probe_m1_ffn import mock
from deepseek_v4_gluon.v41.k2 import k2

PTRS = ["q", "swa", "comp", "kt", "klen", "slot", "swa_idx", "swa_lens", "t2r", "topk", "comp_bt",
        "pos", "sink", "cos_sin", "woa", "woa_s", "wob", "wob_s", "zrec", "res_in", "post_in",
        "comb_in", "pre_in", "hc_fn", "hc_scale", "hc_base", "ffn_w", "res_out", "post_out",
        "comb_out", "pre_out", "gate_w", "bias", "w13", "w13_s", "w2", "w2_s", "sgu", "sgu_s",
        "sw2", "sw2_s", "out", "scratch", "sym", "peers", "epoch", "health"]


def mock_k2(s, tp=4, ratio=1, index=False):
    c = mock(s, tp)
    regions = ["PART", "PSTAT", "SRDY", "XO", "XOS", "CRDY", "X8B", "X8BS", "ARDY", "KRDY"]
    c.update({k: (40 + i) << 20 for i, k in enumerate(regions)})
    c.update(HEADS=64 // tp, RATIO=ratio, INDEX=index, SWA_W=128)
    return c


for s, index in [(int(a), False) for a in sys.argv[1:]] or [(1, False), (16, True), (48, True)]:
    c = mock_k2(s, ratio=4, index=index)
    sig = {n: "*u8" for n in PTRS} | {"swa_stride": "i64", "comp_stride": "i64", "swa_block": "i32",
                                     "comp_block": "i32", "bt_stride": "i32", "qk_scale": "i32",
                                     "rank": "i32", "tl": "i64"} | {k: "constexpr" for k in c}
    r = compile_probe(k2, sig, c, enable_fp_fusion=False)
    a = r["asm"]
    n = lambda p: len(re.findall(p, a))  # noqa: E731
    print(f"S={s} index={index}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"smfma={n('v_mfma_scale_f32_16x16x128')} mfma32={n('v_mfma_f32_16x16x32_bf16')} "
          f"mfma16={n('v_mfma_f32_16x16x16_?bf16')}", flush=True)
