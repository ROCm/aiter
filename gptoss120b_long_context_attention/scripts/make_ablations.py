#!/usr/bin/env python3
"""Write timing-only ablations of the gfx1250 prefill attention kernel (LOOP_VARIANT 2 main loop).

Each variant removes one part of the softmax work in attention_loop_tensor_subtile_split, the loop the
70k prefill runs (BLOCK_M 256, LOOP_VARIANT 2).  Their OUTPUTS ARE WRONG ON PURPOSE: only the time counts,
to see where the time goes before writing a new kernel.
  noexp            the score exps  p = exp2(qk*scale - m)  ->  one FMA  (qk*scale - m) * 0.5 + 1
  norescale        the per-tile accumulator rescale  acc *= alpha  ->  dropped
  noexp_norescale  both
The row-max / alpha / row-sum work stays, so the gap to base is what exp / rescale cost.

  python3 make_ablations.py                      # source: the kernel file aiter imports
  python3 make_ablations.py --src PATH --out DIR
Then time them with:  python3 bench_prefill_attention.py --fp8 --cfg '{"BLOCK_M": 256}' --kernel-file DIR/ua2d_noexp.py
"""
import argparse
import os
import re
import sys

FUNC = "def attention_loop_tensor_subtile_split("
EXP = re.compile(r"gl\.exp2\((qk[01]_shifted)\)")                 # score exps only (alpha's exp stays)
RESCALE = re.compile(r"acc([01]) = acc\1 \* alpha\[:, None\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="", help="unified_attention_2d.py (default: the one aiter imports)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablate"))
    a = ap.parse_args()
    if not a.src:
        import aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d as m
        a.src = m.__file__
    src = open(a.src, encoding="utf-8").read()
    i = src.index(FUNC)
    j = src.index("\n@gluon.jit", i)                                 # end of the loop function
    head, body, tail = src[:i], src[i:j], src[j:]
    n_exp, n_res = len(EXP.findall(body)), len(RESCALE.findall(body))
    print(f"source {a.src}\n  {FUNC[4:-1]}: {n_exp} score exp2 calls, {n_res} accumulator rescales")
    if n_exp == 0 or n_res == 0:
        sys.exit("ERROR: patterns not found; the kernel source differs from the one this script was written for")
    os.makedirs(a.out, exist_ok=True)
    for name, do_exp, do_res in (("base", False, False), ("noexp", True, False),
                                 ("norescale", False, True), ("noexp_norescale", True, True)):
        b = body
        if do_exp:
            b = EXP.sub(r"(\1 * 0.5 + 1.0)", b)
        if do_res:
            b = RESCALE.sub(r"acc\1 = acc\1", b)
        path = os.path.join(a.out, f"ua2d_{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# ABLATION '{name}' of {a.src} -- timing only, outputs are wrong\n" + head + b + tail)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
