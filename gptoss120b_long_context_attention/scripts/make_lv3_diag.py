#!/usr/bin/env python3
"""Split LOOP_VARIANT 3 into its two changes, to see which one costs and which one helps.

Reads the changed kernel file (with attention_loop_lazy_rescale) and writes two kernel files:
  ua2d_foldonly.py  B only: folded fp8 P scale; NO lazy rescale (alpha / M updated every tile, as in
                    LOOP_VARIANT 2: no cross-warp max, no branch around the accumulators)
  ua2d_lazyonly.py  A only: lazy rescale; NO fold (P * 256 multiply before the fp8 cast, as in variant 2;
                    TAU stays 0.75 for fp8 P so P * 256 <= 430 < 448)
Both keep LOOP_VARIANT 3's correctness (accuracy check stays meaningful).  Run them with
  --kernel-file DIR/ua2d_<x>.py --cfg '{"BLOCK_M": 256, "LOOP_VARIANT": 3}'
"""
import argparse
import os
import sys


def sub1(s, a, b, what):
    if s.count(a) != 1:
        sys.exit(f"ERROR: expected exactly one '{what}' in the kernel file, found {s.count(a)}")
    return s.replace(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="aiter/unified_attention_2d_gfx1250.py")
    ap.add_argument("--out", default="lv3_diag")
    a = ap.parse_args()
    src = open(a.src, encoding="utf-8").read()
    if "def attention_loop_lazy_rescale(" not in src:
        sys.exit(f"ERROR: {a.src} has no LOOP_VARIANT 3 (attention_loop_lazy_rescale)")
    os.makedirs(a.out, exist_ok=True)

    # B only: the row-max helper always rescales (LOOP_VARIANT 2 semantics), no reduction, no branch;
    # the accumulator rescales in the loop become unconditional.
    fold = sub1(src,
                "    do_rescale = gl.max((m_new > M + thresh_raw).to(gl.int32), axis=0) > 0\n"
                "    alpha = gl.full(M.shape, 1.0, gl.float32, layout=M.type.layout)\n"
                "    if do_rescale:\n"
                "        alpha = gl.exp2(M * QK_scale - m_new * QK_scale)\n"
                "        M = m_new\n",
                "    do_rescale = True  # DIAG foldonly: always rescale (no cross-warp max, no branch)\n"
                "    alpha = gl.exp2(M * QK_scale - m_new * QK_scale)\n"
                "    M = m_new\n",
                "lazy decision in _lazy_row_max")
    i = fold.index("def attention_loop_lazy_rescale(")
    j = fold.index("\n@gluon.jit", i)
    body = fold[i:j]
    n = body.count("if do_rescale:")
    body = body.replace("if do_rescale:  # lazy rescale (A): skipped on most tiles", "if True:  # DIAG foldonly")
    body = body.replace("if do_rescale:", "if True:  # DIAG foldonly")
    fold = fold[:i] + body + fold[j:]
    print(f"foldonly: lazy decision removed, {n} accumulator branches made unconditional")

    # A only: no fold (variant-2 cast and unscaled exponent), lazy rescale kept with the fp8 threshold.
    lazy = sub1(src,
                "    FOLD: gl.constexpr = q.dtype.is_fp8()\n",
                "    FOLD: gl.constexpr = False  # DIAG lazyonly: no folded fp8 scale\n",
                "FOLD definition")
    lazy = sub1(lazy,
                "        TAU: gl.constexpr = _LAZY_TAU\n",
                "        if q.dtype.is_fp8():  # DIAG lazyonly: fp8 P without the fold still needs TAU 0.75\n"
                "            TAU: gl.constexpr = _LAZY_TAU_FP8\n"
                "        else:\n"
                "            TAU: gl.constexpr = _LAZY_TAU\n",
                "bf16 TAU")
    print("lazyonly: fold removed (P * 256 before the fp8 cast), lazy rescale kept (TAU 0.75 for fp8 P)")

    for name, text in (("foldonly", fold), ("lazyonly", lazy)):
        path = os.path.join(a.out, f"ua2d_{name}.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# LOOP_VARIANT 3 DIAGNOSTIC '{name}' of {a.src}\n" + text)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
