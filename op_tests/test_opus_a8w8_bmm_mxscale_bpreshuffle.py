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
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw
from aiter.ops.shuffle import (
    shuffle_scale_a,
    shuffle_scale_b,
    shuffle_scale_mxsk_mpack,
    shuffle_weight,
)
from aiter.test_common import run_perftest
from test_opus_a8w8_bmm import (
    GROUP,
    _block_varied,
    _preshuffled_kids,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    run_torch,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# Taken from the catalog rather than written out, because a kid that is not in
# the built .so does not fail -- the dispatch falls through and returns another
# kernel's answer. A hardcoded list that outlived two of its kids reported those
# two at 1.41 relative error, which looks like a kernel bug and is not one.
KIDS_BPRESHUFFLE = tuple(sorted(_preshuffled_kids()))
KID_PLAIN = 320  # same tile, row-major B, broadcast scale pack


def _mpack_kids():
    """kid -> (B_M, SFA_MB) for the kids wanting a host M-packed A scale.

    Same hazard as the preshuffled weight: the packed panel is a permutation of
    the same bytes, so a kid handed the plain scale runs and answers wrongly.
    Read from the catalog for the same reason the kid list is.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): inst.needs_mpacked_sfa
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
        if inst.needs_mpacked_sfa
    }


def _shuffle_scale_kids():
    """kid -> shuffle_scale_a's ``sub`` for the kids wanting that layout.

    Same hazard as _mpack_kids, and read from the catalog for the same reason.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): inst.needs_shuffle_scale
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
        if inst.needs_shuffle_scale
    }


MPACK_KIDS = _mpack_kids()
SHUFFLE_SCALE_KIDS = _shuffle_scale_kids()
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

    _sfa = {}

    def _sfa_for(kid):
        sub = SHUFFLE_SCALE_KIDS.get(kid)
        if sub:
            if kid not in _sfa:
                # stride(0) is zeroed: the shuffle_scale layout folds the row into its own
                # addressing, so the kernel takes only the per-batch slab from
                # stride(1) and would otherwise add a bogus row offset.
                slab = shuffle_scale_a(xs_mx, k, sub)
                _sfa[kid] = slab.as_strided(
                    (m, g, slab.shape[1]), (0, slab.shape[1], 1)
                )
            return _sfa[kid]
        mpack = MPACK_KIDS.get(kid)
        if not mpack:
            return xs_in
        if kid not in _sfa:
            sk = xs_mx.shape[2]
            _sfa[kid] = (
                shuffle_scale_mxsk_mpack(xs_mx, *mpack).view(g, -1, sk).transpose(0, 1)
            )
        return _sfa[kid]

    ws_shuf = None

    def _sfb_for(kid):
        nonlocal ws_shuf
        if kid not in SHUFFLE_SCALE_KIDS:
            return ws_mx
        if ws_shuf is None:
            # Kept 3-D with the N-block axis in the middle so stride(0) is the
            # per-batch slab, which is the only term the shuffle_scale path reads.
            ws_shuf = shuffle_scale_b(ws_mx, n, k).view(g, n // 128, -1)
        return ws_shuf

    def _call(kid, W):
        Y = torch.zeros((m, g, n), dtype=ydt)
        _opus_bmm_a8w8_mxscale_raw(
            O_in, W, Y, _sfa_for(kid), _sfb_for(kid), split_k, kid
        )
        torch.cuda.synchronize()
        return Y

    # A kid whose tile does not divide this shape rejects the call rather than
    # returning something wrong -- e.g. the B_K=512 tiles need more K-tiles than
    # K=1024 provides. Those are absent from the row, not failures.
    errs = {}
    skipped = []
    for kid in KIDS_BPRESHUFFLE:
        try:
            errs[kid] = _rel_err(_call(kid, W_sh), ref)
        except RuntimeError:
            skipped.append(kid)
    err_plain = _rel_err(_call(KID_PLAIN, W_mx), ref)

    # The public entry end to end: guarded custom op -> tuned row -> a kid that
    # wants the shuffled weight. Only a shape the active CSV covers with such a
    # kid can run it (b_preshuffled=True raises by design otherwise), so point
    # AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE at the preshuffle table
    # to exercise it; an uncovered shape is reported as off, not failed.
    err_pub = None
    if split_k == 1:  # the entry defaults splitK, so only compare where they agree
        try:
            err_pub = _rel_err(
                batched_gemm_a8w8_mxscale(
                    O_in, W_sh, xs_in, ws_mx, dtype=ydt, b_preshuffled=True
                ),
                ref,
            )
        except ValueError:
            pass

    row = f"g={g:<3} m={m:<6} n={n:<6} k={k:<6} sk={split_k} "
    row += "  ".join(f"kid{kid}={e:.5f}" for kid, e in errs.items())
    row += f"  kid{KID_PLAIN}={err_plain:.5f}"
    row += f"  public={'off' if err_pub is None else f'{err_pub:.5f}'}"
    if skipped:
        row += "  skipped=" + ",".join(str(k) for k in skipped)
    if bench:

        def _time(kid, W):
            _, t = run_perftest(
                _opus_bmm_a8w8_mxscale_raw,
                O_in,
                W,
                torch.zeros((m, g, n), dtype=ydt),
                _sfa_for(kid),
                _sfb_for(kid),
                split_k,
                kid,
                num_warmup=5,
            )
            return t

        times = [f"kid{kid}={_time(kid, W_sh):.2f}us" for kid in errs]
        times.append(f"kid{KID_PLAIN}={_time(KID_PLAIN, W_mx):.2f}us")
        for kid in KIDS_PLAIN_EXTRA:
            times.append(f"kid{kid}={_time(kid, W_mx):.2f}us")
        row += "  |  " + "  ".join(times)
    print(row, flush=True)
    # The preshuffle kids differ from the plain one only in where B's bytes live
    # and how the (identical) scale bytes reach the MFMA, so they must land on
    # the plain kid's accuracy, not merely "close to" the reference.
    tol = max(2.0 * err_plain, err_plain + 1e-4)
    ok = all(e <= tol for e in errs.values())
    return ok and (err_pub is None or err_pub <= tol)


def _check_dispatch():
    """b_preshuffled routing at the public entry, without launching anything.

    None of these outcomes shows up in the output tensor -- a kid mismatched to
    B's layout returns a plausible wrong answer rather than failing -- so this
    spies on the raw binding to pin down which kid each combination resolves to,
    on meta tensors. It drives the unwrapped impl because the public entry is a
    registered custom op whose meta kernel would answer instead of the dispatch,
    and it feeds the tuned row in directly so the checks do not depend on which
    CSV the environment happens to point at.
    """
    from unittest.mock import patch

    import aiter.ops.batched_gemm_op_a8w8 as bg
    import aiter.ops.opus.bmm_op as bmm

    g, m, n, k = 2, 128, 1024, 4096
    pre = bmm._mxscale_kid_pre_b()
    kid_pre = next(
        kid
        for kid in KIDS_BPRESHUFFLE
        if pre.get(kid) and bmm._kid_takes_plain_scales(kid)
    )
    kid_host_scale = min(bmm._mxscale_kid_host_scale())

    args = (
        torch.empty((m, g, k), dtype=dtypes.fp8, device="meta"),
        torch.empty((g, n, k), dtype=dtypes.fp8, device="meta"),
        torch.empty((m, g, k // GROUP), dtype=torch.uint8, device="meta"),
        torch.empty((g, n // GROUP, k // GROUP), dtype=torch.uint8, device="meta"),
    )

    def _resolve(row, b_preshuffled):
        """The kid this tuned row dispatches to, or the ValueError it raises."""
        seen = {}

        def _spy(x, wo_a, Y, sfa, sfb, splitK, kernelId):
            seen["kid"] = int(kernelId)

        with patch.object(bmm, "_opus_bmm_a8w8_mxscale_raw", _spy), patch.object(
            bg, "lookup_mxscale_bmm_config", lambda *a, **kw: row
        ):
            try:
                bg._batched_gemm_a8w8_mxscale_impl(*args, b_preshuffled=b_preshuffled)
            except ValueError as err:
                return err
        return seen["kid"]

    def _row(kid):
        return {"libtype": "opus", "kernelId": kid, "splitK": 1}

    def _row_major_kid(r):
        return isinstance(r, int) and not pre.get(r)

    cases = (
        (
            f"tuned kid{kid_pre} + declared preshuffled -> runs it",
            _row(kid_pre),
            True,
            lambda r: r == kid_pre,
        ),
        (
            f"tuned kid{kid_pre} + row-major B -> row-major fallback",
            _row(kid_pre),
            False,
            _row_major_kid,
        ),
        (
            f"tuned kid{KID_PLAIN} (row-major) + declared preshuffled -> raises",
            _row(KID_PLAIN),
            True,
            lambda r: isinstance(r, ValueError),
        ),
        (
            f"tuned kid{kid_host_scale} (host-rearranged scales) + declared -> raises",
            _row(kid_host_scale),
            True,
            lambda r: isinstance(r, ValueError),
        ),
        (
            "no tuned row + declared preshuffled -> raises",
            None,
            True,
            lambda r: isinstance(r, ValueError),
        ),
        ("no tuned row + row-major B -> heuristic", None, False, _row_major_kid),
    )

    ok = True
    for label, row, b_preshuffled, want in cases:
        got = _resolve(row, b_preshuffled)
        good = want(got)
        ok &= good
        shown = "ValueError" if isinstance(got, ValueError) else f"kid{got}"
        print(f"  {'ok  ' if good else 'FAIL'} {label}  [{shown}]", flush=True)
    return ok


def main():
    global KIDS_BPRESHUFFLE, KIDS_PLAIN_EXTRA
    p = argparse.ArgumentParser()
    p.add_argument("-g", type=int, default=2, help="batch (group) count")
    p.add_argument("-n", type=int, default=1024, help="N (multiple of 32)")
    p.add_argument("-k", type=int, default=4096, help="K (multiple of 256)")
    p.add_argument(
        "-s",
        "--sizes",
        default="64,128,129,256,1024",
        help="comma-separated M list (M needs no alignment: partial tiles are masked)",
    )
    p.add_argument("-d", "--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--bench", action="store_true", help="also time every kid")
    p.add_argument(
        "--split-k",
        type=int,
        default=1,
        help="splitK (>1 routes through the fp32 workspace + reduce)",
    )
    p.add_argument(
        "--kids",
        default=None,
        help=f"comma-separated preshuffle kids (default {','.join(map(str, KIDS_BPRESHUFFLE))})",
    )
    p.add_argument(
        "--extra-kids",
        default=None,
        help=f"comma-separated row-major kids to time alongside "
        f"(default {','.join(map(str, KIDS_PLAIN_EXTRA))})",
    )
    args = p.parse_args()

    if args.kids is not None:
        KIDS_BPRESHUFFLE = tuple(int(x) for x in args.kids.split(",") if x)
    if args.extra_kids is not None:
        KIDS_PLAIN_EXTRA = tuple(int(x) for x in args.extra_kids.split(",") if x)

    if get_gfx() not in SUPPORTED_GFX:
        print(f"skip: {get_gfx()} not in {SUPPORTED_GFX}")
        return 0

    ydt = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp32
    assert args.n % 32 == 0, "these kids tile N by 32"
    assert args.k % 256 == 0, "these kids tile K by 256"
    assert args.k % GROUP == 0

    print("dispatch: b_preshuffled routing at the public entry", flush=True)
    ok = _check_dispatch()
    for m in [int(x) for x in args.sizes.split(",")]:
        ok &= _run(args.g, m, args.n, args.k, ydt, args.bench, args.split_k)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
