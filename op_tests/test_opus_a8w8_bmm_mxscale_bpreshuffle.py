# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf check for the opus fp8 e8m0 mxscale BMM B-preshuffle kids.

kid 170 is kid 320's tile (64x32x256 -> a 32x32x256 per-wave register tile, i.e.
COM_REP_M/N/K = 2/2/2) with two compile-time axes flipped: the weight comes from
``shuffle_weight(w, layout=(16, 16))`` and every MFMA picks its e8m0 byte with
the hardware ``scale_op_sel`` immediate instead of a broadcast pack. kid 171 adds
a third: since the preshuffle order already is the mfma_16x16x128 B fragment
order, its consumer waves ``buffer_load`` B straight into the MFMA registers and
B never touches LDS.

All kids run on the same quantized data -- kid 320 on the row-major weight, 170
and 171 on the preshuffled one -- so a mismatch localizes to the preshuffle /
op_sel / direct-B changes rather than to the quantization or the reference.

Usage:
    python3 op_tests/test_opus_a8w8_bmm_mxscale_bpreshuffle.py
    python3 op_tests/test_opus_a8w8_bmm_mxscale_bpreshuffle.py -g 4 -n 1024 -k 4096
"""

import argparse
import sys

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest
from test_opus_a8w8_bmm import (
    GROUP,
    _block_varied,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# Both preshuffle kids share kid320's 64x32x256 tile; 171 additionally skips LDS
# for B (consumers buffer_load their MFMA B fragments straight from global).
KIDS_BPRESHUFFLE = (170, 171, 178, 179)
KID_PLAIN = 320  # same tile, row-major B, broadcast scale pack
# Extra row-major kids to time alongside, e.g. whichever the tuner actually
# ships for the shape under test (kid158 is the large-M pick on many of them).
KIDS_PLAIN_EXTRA = (311, 321, 653, 325, 158)


def _rel_err(y, ref):
    return (y.float() - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)


def _run(g, m, n, k, ydt, bench, split_k=1):
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)
    # Same bytes, 16x16-tiled: [G, N, K] -> [G][N/16][K/32][2][16 n][16 k].
    W_sh = shuffle_weight(W_mx, layout=(16, 16))

    O_in = O_mx.transpose(0, 1)  # [m, g, k] mmajor view
    xs_in = xs_mx.transpose(0, 1)  # [m, g, k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)  # [m, g, n]

    def _call(kid, W):
        Y = torch.zeros((m, g, n), dtype=ydt)
        _opus_bmm_a8w8_mxscale_raw(O_in, W, Y, xs_in, ws_mx, split_k, kid)
        torch.cuda.synchronize()
        return Y

    errs = {kid: _rel_err(_call(kid, W_sh), ref) for kid in KIDS_BPRESHUFFLE}
    err_plain = _rel_err(_call(KID_PLAIN, W_mx), ref)

    row = f"g={g:<3} m={m:<6} n={n:<6} k={k:<6} sk={split_k} "
    row += "  ".join(f"kid{kid}={e:.5f}" for kid, e in errs.items())
    row += f"  kid{KID_PLAIN}={err_plain:.5f}"
    if bench:
        def _time(kid, W):
            _, t = run_perftest(
                _opus_bmm_a8w8_mxscale_raw,
                O_in, W, torch.zeros((m, g, n), dtype=ydt), xs_in, ws_mx,
                split_k, kid, num_warmup=5,
            )
            return t
        times = [f"kid{kid}={_time(kid, W_sh):.2f}us" for kid in KIDS_BPRESHUFFLE]
        times.append(f"kid{KID_PLAIN}={_time(KID_PLAIN, W_mx):.2f}us")
        for kid in KIDS_PLAIN_EXTRA:
            times.append(f"kid{kid}={_time(kid, W_mx):.2f}us")
        row += "  |  " + "  ".join(times)
    print(row, flush=True)
    # The preshuffle kids differ from the plain one only in where B's bytes live
    # and how the (identical) scale bytes reach the MFMA, so they must land on
    # the plain kid's accuracy, not merely "close to" the reference.
    tol = max(2.0 * err_plain, err_plain + 1e-4)
    return all(e <= tol for e in errs.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g", type=int, default=2, help="batch (group) count")
    p.add_argument("-n", type=int, default=1024, help="N (multiple of 32)")
    p.add_argument("-k", type=int, default=4096, help="K (multiple of 256)")
    p.add_argument(
        "-s", "--sizes", default="64,128,129,256,1024",
        help="comma-separated M list (M needs no alignment: partial tiles are masked)",
    )
    p.add_argument("-d", "--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--bench", action="store_true", help="also time every kid")
    p.add_argument(
        "--split-k", type=int, default=1,
        help="splitK (>1 routes through the fp32 workspace + reduce)",
    )
    args = p.parse_args()

    if get_gfx() not in SUPPORTED_GFX:
        print(f"skip: {get_gfx()} not in {SUPPORTED_GFX}")
        return 0

    ydt = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp32
    assert args.n % 32 == 0, "these kids tile N by 32"
    assert args.k % 256 == 0, "these kids tile K by 256"
    assert args.k % GROUP == 0

    ok = True
    for m in [int(x) for x in args.sizes.split(",")]:
        ok &= _run(args.g, m, args.n, args.k, ydt, args.bench, args.split_k)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
