# SPDX-License-Identifier: MIT
"""CPU: compile the V4.1 K1 launch (attention seam + front), mock region offsets."""
import re
import sys

from deepseek_v4_gluon.common.probe import compile_probe
from deepseek_v4_gluon.v41.k1 import k1

PTRS = ["res_in", "pend", "post_in", "comb_in", "pre_in", "hc_fn", "hc_scale", "hc_base", "attn_w",
        "res_out", "post_out", "comb_out", "pre_out", "wqkv", "wqkv_s", "qn_w", "kvn_w", "wqb",
        "wqb_s", "cos_sin", "pos", "slot", "swa", "q", "swa_idx", "swa_lens", "comp_bt", "topk",
        "t2r", "kt", "klen", "normed_out", "qr_out", "scratch", "epoch", "health"]


def mock_k1(s, tp=4, fold=True):
    regions = ["LIN", "PMIX", "X8", "X8S", "XRDY", "QKVP", "QX8", "QX8S", "QRDY"]
    c = {k: (i + 1) << 20 for i, k in enumerate(regions)}
    c.update(S=s, TP=tp, HIDDEN=5120, RATIO=4, FOLD=fold, INDEX=False, SWA_W=128)
    return c


for s, tp in [(int(a), 4) for a in sys.argv[1:]] or [(1, 4), (16, 4), (48, 4), (16, 2)]:
    c = mock_k1(s, tp)
    sig = {n: "*u8" for n in PTRS} | {"swa_stride": "i64", "swa_block": "i32", "comp_block": "i32",
                                     "bt_stride": "i32", "tl": "i64"} | {k: "constexpr" for k in c}
    r = compile_probe(k1, sig, c, enable_fp_fusion=False)
    a = r["asm"]
    n = lambda p: len(re.findall(p, a))  # noqa: E731
    print(f"tp={tp} S={s}: vgpr={r['vgpr']} spills={r['spills']} lds={r['lds']} "
          f"smfma={n('v_mfma_scale_f32_16x16x128')} f32mfma={n('v_mfma_f32_16x16x4')}", flush=True)
