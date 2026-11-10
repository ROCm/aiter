# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The trained A16W16 fallback selector on untuned shapes: result check and time against
today's default chain.

Per shape: the selector's pick through tgemm.mm (result checked against an fp32 torch
reference), then the pick and the default chain timed the same way (rotated input copies,
one warm-up pass over them, 20 CUDA-event-timed calls, median). The tuned CSV is ignored
(--tuned keeps it), so every shape is treated as untuned.

  python3 op_tests/test_gemm_a16w16_selector.py                  # one shape per backend
  python3 op_tests/test_gemm_a16w16_selector.py -mnk 1,4096,4096 64,8192,1024
"""

import argparse
import os
import tempfile

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("-mnk", nargs="*", default=None, help="M,N,K shapes")
parser.add_argument("--tuned", action="store_true", help="keep aiter's tuned CSV rows")
parser.add_argument("--csv", default=None, help="write the per-shape table here")
args = parser.parse_args()

os.environ.setdefault("AITER_GEMM_A16W16_SELECTOR", "1")
os.environ.setdefault("AITER_GEMM_A16W16_SELECTOR_CACHE", "0")
if not args.tuned:
    empty = tempfile.NamedTemporaryFile("w", suffix="_bf16_tuned_gemm.csv", delete=False)
    empty.write("gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,"
                "us,kernelName,err_ratio,tflops,bw\n")
    empty.close()
    os.environ["AITER_CONFIG_GEMM_BF16"] = empty.name

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aiter import tuned_gemm  # noqa: E402
from aiter.ops import gemm_a16w16_selector as gs  # noqa: E402
from aiter.test_common import checkAllclose  # noqa: E402

# One audit shape (an (N, K) pair the shipped artifact never trained on) per backend the
# selector picks there.
DEFAULT_SHAPES = [(2, 64, 7168), (8, 2048, 2560), (512, 8192, 256), (2048, 100, 5120),
                  (2, 14336, 8192), (32, 4096, 352), (16, 5376, 512), (128, 1280, 2880),
                  (8192, 12800, 5120), (2048, 4096, 4096)]


def config(m, n, k, use_selector):
    os.environ[gs.ENV] = "1" if use_selector else "0"
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    bf16 = str(torch.bfloat16)
    return tuned_gemm.get_GEMM_A16W16_config(m, n, k, False, bf16, bf16, False, False)


def main():
    shapes = ([tuple(map(int, s.split(","))) for s in args.mnk] if args.mnk else DEFAULT_SHAPES)
    rows = []
    for m, n, k in shapes:
        dflt = config(m, n, k, False)
        pick = config(m, n, k, True)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = tuned_gemm.tgemm.mm(a, b)
        ref = torch.nn.functional.linear(a.float(), b.float())
        err = checkAllclose(ref, out.float(), rtol=5e-2, atol=5e-2,
                            msg=f"({m}, {n}, {k}) {pick.get('selector', pick['libtype'])}: ")
        _, us = gs._trial([pick, dflt], m, n, k, tuned_gemm._run_config, check=False)
        name = lambda c: c.get("selector") or f"{c['libtype']}|default"
        rows.append({"M": m, "N": n, "K": k, "pick": name(pick), "default": name(dflt),
                     "pick_us": us[0], "default_us": us[1],
                     "speedup": us[1] / us[0] if us[0] and us[1] else None, "err_ratio": err})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ok = df.err_ratio <= 0.05
    sp = df.speedup.dropna()
    print(f"\n{int(ok.sum())} of {len(df)} picks pass the result check; geomean speedup over the "
          f"default chain {float((sp.prod()) ** (1 / len(sp))):.3f}x" if len(sp) else "")
    if args.csv:
        df.to_csv(args.csv, index=False)
    assert ok.all(), "a selector pick failed the result check"
    check_pick_cache(shapes[0])


def check_pick_cache(shape):
    """A timed top-2 pick is written to the pick cache, and a fresh selector reuses it
    without running a kernel; a stamp change empties the file."""
    m, n, k = shape
    bf16 = str(torch.bfloat16)
    gfx, cu = tuned_gemm.get_gfx(), tuned_gemm.get_cu_num()
    path = os.path.join(tempfile.mkdtemp(), "picks.csv")
    os.environ.update({gs.ENV: "1", gs.ENV_CACHE: path, gs.ENV_TOPK: "2"})

    def select(run):
        return gs.select(m, n, k, False, bf16, bf16, False, False, gfx, cu, run)

    def no_run(*_):
        raise AssertionError("a cached pick ran a kernel")

    try:
        gs.get_selector.cache_clear()
        first = select(tuned_gemm._run_config)
        gs.get_selector.cache_clear()
        again = select(no_run)
        assert again == first, (first, again)
        os.environ[gs.ENV_TOPK] = "3"
        gs.get_selector.cache_clear()
        assert len(gs.get_selector(gfx, cu).cache.rows) == 1
        with open(path) as f:
            lines = f.read().splitlines()
        with open(path, "w") as f:
            f.write("\n".join(["# {}"] + lines[1:]) + "\n")
        gs.get_selector.cache_clear()
        assert not gs.get_selector(gfx, cu).cache.rows, "a stale stamp kept its rows"
        print(f"pick cache: ({m}, {n}, {k}) -> {first and first['selector']} reused from {path}")
    finally:
        for v in (gs.ENV_CACHE, gs.ENV_TOPK):
            os.environ.pop(v, None)
        gs.get_selector.cache_clear()


if __name__ == "__main__":
    main()
