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
import os
import sys

import torch
from test_opus_a8w8_bmm import (
    GROUP,
    _block_varied,
    _preshuffled_kids,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    _scale_picker,
    run_torch,
)

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.opus import policy
from aiter.ops.opus.gemm_op_a8w8 import (
    _opus_gemm_a8w8_mxscale_bmm_launch_raw,
    _mxscale_bmm_workspace,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# Taken from the catalog rather than written out, because a kid that is not in
# the built .so does not fail -- the dispatch falls through and returns another
# kernel's answer. A hardcoded list that outlived two of its kids reported those
# two at 1.41 relative error, which looks like a kernel bug and is not one.
KIDS_BPRESHUFFLE = tuple(sorted(_preshuffled_kids()))
# --kids narrows KIDS_BPRESHUFFLE to whatever the caller wants timed, but the
# dispatch self-check is about the catalogue, not about that subset: picking its
# probe kid out of a one-element list raised StopIteration and failed the run
# before any kernel ran.
ALL_KIDS_BPRESHUFFLE = KIDS_BPRESHUFFLE
KID_PLAIN = 8320  # same tile, row-major B, broadcast scale pack


# Extra row-major kids to time alongside, e.g. whichever the tuner actually
# ships for the shape under test (kid8158 is the large-M pick on many of them).
KIDS_PLAIN_EXTRA = (8311, 8321, 8653, 8325, 8158)


def _rel_err(y, ref):
    return (y.float() - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)


def _run(g, m, n, k, ydt, bench, split_k=1, group=GROUP):
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16, group=group)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16, group=group)
    # Same bytes, 16x16-tiled: [G, N, K] -> [G][N/16][K/32][2][16 n][16 k].
    W_sh = shuffle_weight(W_mx, layout=(16, 16))

    O_in = O_mx.transpose(0, 1)  # [m, g, k] mmajor view
    xs_in = xs_mx.transpose(0, 1)  # [m, g, k/group] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32, group=group).transpose(
        0, 1
    )  # [m, g, n]

    scale_for = _scale_picker(xs_mx, ws_mx, n, k)

    def _call(kid, W):
        Y = torch.zeros((m, g, n), dtype=ydt)
        xs_kid, ws_kid = scale_for(kid)
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(
            O_in,
            W,
            Y,
            xs_kid,
            ws_kid,
            workspace=_mxscale_bmm_workspace(O_in, W, Y, kid, split_k),
            kid=kid,
            split_k=split_k,
        )
        torch.cuda.synchronize()
        return Y

    # A kid whose tile does not divide this shape rejects the call rather than
    # returning something wrong -- e.g. the B_K=512 tiles need more K-tiles than
    # K=1024 provides. Those are absent from the row, not failures.
    errs = {}
    skipped = []
    kid_plain = KID_PLAIN + (1000 if group == 32 else 0)
    for kid in KIDS_BPRESHUFFLE:
        if policy.mxscale_bmm_kid_group(kid) != group:
            continue
        try:
            errs[kid] = _rel_err(_call(kid, W_sh), ref)
        except (RuntimeError, ValueError):
            skipped.append(kid)
    # The row-major reference kid, or at a K too short for it the 128-deep one.
    for cand in (kid_plain, kid_plain - KID_PLAIN + 8653):
        try:
            err_plain = _rel_err(_call(cand, W_mx), ref)
        except RuntimeError as e:
            plain_exc = e
            continue
        kid_plain = cand
        break
    else:
        raise plain_exc

    # The public entry end to end: guarded custom op -> the preshuffle table
    # (the entry picks that one, nothing to set) -> the row's backend, opus or
    # flydsl, or flydsl's heuristic for a shape the table lacks.
    err_pub = None
    if split_k == 1:  # the entry defaults splitK, so only compare where they agree
        err_pub = _rel_err(
            batched_gemm_a8w8_mxscale_bpreshuffle(
                O_in.contiguous(), W_sh, xs_in.contiguous(), ws_mx, dtype=ydt
            ),
            ref,
        )

    row = f"G{group:<3} g={g:<3} m={m:<6} n={n:<6} k={k:<6} sk={split_k} "
    row += "  ".join(f"kid{kid}={e:.5f}" for kid, e in errs.items())
    row += f"  kid{kid_plain}={err_plain:.5f}"
    row += f"  public={'off' if err_pub is None else f'{err_pub:.5f}'}"
    if skipped:
        row += "  skipped=" + ",".join(str(k) for k in skipped)
    if bench:

        def _time(kid, W):
            xs_kid, ws_kid = scale_for(kid)
            _, t = run_perftest(
                _opus_gemm_a8w8_mxscale_bmm_launch_raw,
                O_in,
                W,
                torch.zeros((m, g, n), dtype=ydt),
                xs_kid,
                ws_kid,
                # Keyword from here on: the unified entry's tail is
                # (workspace, kid, split_k), the reverse of the old
                # (split_k, kid), and positionally both orders type-check.
                workspace=None,
                kid=kid,
                split_k=split_k,
                num_warmup=5,
            )
            return t

        times = [f"kid{kid}={_time(kid, W_sh):.2f}us" for kid in errs]
        times.append(f"kid{kid_plain}={_time(kid_plain, W_mx):.2f}us")
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


def _check_tables():
    """One tuned table per B layout, holding only kids the entry can dispatch.

    Keeping the tables apart is what keeps the row-major caller tuned: a shared
    table would hand it rows naming preshuffled kids, every one of which it has
    to drop for the heuristic, silently losing the shipped table's pick. So this
    reads both tables the way the entry does and checks no row crosses over --
    which also catches a preshuffle CSV whose name lets the shipped table's glob
    merge it in.

    The scale layout is checked on the same rows because it is the other way a
    tuned row can name a kid the entry refuses. A tuner sweeping every codegen
    instance sees the 8 that want their scales rearranged on the host, and they
    do win cells; crowned there, the row is dead weight (b_preshuffled=False) or
    raises (True). Both guards live in the backend, so a table that trips one is
    a tuning bug, not a dispatch bug -- catch it here rather than in a serving
    log.
    """
    from aiter.jit.core import AITER_CONFIGS

    gfx = get_gfx()
    ok = True
    for b_preshuffled in (False, True):
        path = (
            AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE_FILE
            if b_preshuffled
            else AITER_CONFIGS.AITER_CONFIG_BATCHED_GEMM_A8W8_BLOCKSCALE_MXSCALE_FILE
        )
        rows = policy._load_mxscale_bmm_tuned(None, b_preshuffled)
        # kernelId is only opus's to interpret, and the catalog behind the check
        # is this arch's.
        kids = {
            int(row["kernelId"])
            for key, row in rows.items()
            if key[0] == gfx and row.get("libtype") == "opus"
        }
        # The key's last field is the row's w_scale_block; a kid of the other
        # block would read the scales at the wrong stride.
        bad_g = sorted(
            {
                int(row["kernelId"])
                for key, row in rows.items()
                if key[0] == gfx
                and row.get("libtype") == "opus"
                and policy.mxscale_bmm_kid_group(int(row["kernelId"]))
                != policy.mxscale_bmm_group_of_block(key[5])
            }
        )
        bad_b = sorted(
            k
            for k in kids
            if not policy.mxscale_bmm_kid_takes_b_layout(k, b_preshuffled)
        )
        bad_sf = sorted(
            k for k in kids if not policy.mxscale_bmm_kid_takes_plain_scales(k)
        )
        bad = bool(bad_b or bad_sf or bad_g)
        ok &= not bad
        label = f"b_preshuffled={b_preshuffled!s:<5} -> {len(kids)} kid(s)"
        if bad_b:
            note = f"wrong B layout: {bad_b}"
        elif bad_sf:
            note = f"wants host-rearranged scales: {bad_sf}"
        elif bad_g:
            note = f"w_scale_block disagrees with the kid: {bad_g}"
        else:
            note = os.path.basename(path)
        print(f"  {'FAIL' if bad else 'ok  '} {label}  [{note}]", flush=True)
    return ok


def _check_dispatch():
    """B-layout routing across the two public entries, without launching anything.

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
    import aiter.ops.opus.gemm_op_a8w8 as bmm

    g, m, n, k = 2, 128, 1024, 4096
    # kid -> (m_align, needs_preshuffled_b, needs_host_rearranged_scales, GROUP_K)
    table = policy._mxscale_bmm_kid_table()
    pre = {kid: entry[1] for kid, entry in table.items()}
    kid_pre = next(
        kid
        for kid in ALL_KIDS_BPRESHUFFLE
        if pre.get(kid)
        and policy.mxscale_bmm_kid_takes_plain_scales(kid)
        and policy.mxscale_bmm_kid_group(kid) == 128
        and policy.mxscale_bmm_kid_group(kid + 1000) == 32
    )
    kid_pre32 = kid_pre + 1000
    # Empty unless the catalogue builds the relaid-scale kids
    # (BMM_BUILD_RELAID_SCALE_KIDS); their refusal case is skipped then.
    host_scale_kids = sorted(kid for kid, entry in table.items() if entry[2])

    def _args(group, n_=n):
        return (
            torch.empty((m, g, k), dtype=dtypes.fp8, device="meta"),
            torch.empty((g, n_, k), dtype=dtypes.fp8, device="meta"),
            torch.empty((m, g, k // group), dtype=torch.uint8, device="meta"),
            torch.empty(
                (g, max(1, n_ // group), k // group), dtype=torch.uint8, device="meta"
            ),
        )

    def _resolve(row, b_preshuffled, group=128, n_=n):
        """The kid this tuned row dispatches to, or the ValueError it raises."""
        seen = {}

        # Mirrors the unified entry exactly, keywords included: the production
        # call passes kid/split_k by name, so a spy still shaped like the old
        # (splitK, kernelId) positional tail would raise TypeError here rather
        # than record anything.
        def _spy(x, wo_a, Y, sfa, sfb, workspace=None, kid=0, split_k=1):
            seen["kid"] = int(kid)

        def _spy_flydsl(x, wo_a, x_scale, w_scale, out, kernel_name=None, **_kw):
            seen["kid"] = "flydsl"
            return out

        impl = (
            bg._batched_gemm_a8w8_mxscale_bpreshuffle_impl
            if b_preshuffled
            else bg._batched_gemm_a8w8_mxscale_impl
        )
        # The row-major entry caches both the resolved launcher and the plan,
        # which would otherwise keep the first case's spy and row.
        caches = (bg._get_mxscale_bmm_launchers, bg._get_mxscale_bmm_launch_plan)
        for c in caches:
            c.cache_clear()
        import aiter.ops.flydsl.batched_gemm_a8w8 as fly

        with patch.object(
            bmm, "_opus_gemm_a8w8_mxscale_bmm_launch_raw", _spy
        ), patch.object(fly, "run_bmm_a8w8_mxfp8", _spy_flydsl), patch.object(
            policy, "lookup_mxscale_bmm_config", lambda *a, **kw: row
        ):
            try:
                # The w_scale shape carries the block the entry reads.
                impl(*_args(group, n_))
            except ValueError as err:
                return err
            finally:
                for c in caches:
                    c.cache_clear()
        return seen["kid"]

    def _row(kid):
        return {"libtype": "opus", "kernelId": kid, "splitK": 1}

    def _is_flydsl(r):
        return r == "flydsl"

    def _row_major_kid(group):
        return lambda r: (
            isinstance(r, int)
            and not pre.get(r)
            and policy.mxscale_bmm_kid_group(r) == group
        )

    def _pre_fallback(group, not_kid=None):
        return lambda r: (
            isinstance(r, int)
            and bool(pre.get(r))
            and policy.mxscale_bmm_kid_takes_plain_scales(r)
            and policy.mxscale_bmm_kid_group(r) == group
            and r != not_kid
        )

    # (label, tuned row, b_preshuffled, scale group, N, expectation)
    cases = (
        (
            f"tuned kid{kid_pre} + declared preshuffled -> runs it",
            _row(kid_pre),
            True,
            128,
            n,
            lambda r: r == kid_pre,
        ),
        (
            f"tuned kid{kid_pre} + row-major B -> row-major fallback",
            _row(kid_pre),
            False,
            128,
            n,
            _row_major_kid(128),
        ),
        (
            f"tuned kid{KID_PLAIN} (row-major) + declared preshuffled -> preshuffled fallback",
            _row(KID_PLAIN),
            True,
            128,
            n,
            _pre_fallback(128),
        ),
        *(
            (
                (
                    (
                        f"tuned kid{host_scale_kids[0]} (host-rearranged scales) "
                        "+ declared -> preshuffled fallback"
                    ),
                    _row(host_scale_kids[0]),
                    True,
                    128,
                    n,
                    _pre_fallback(128, host_scale_kids[0]),
                ),
            )
            if host_scale_kids
            else ()
        ),
        (
            "no tuned row + declared preshuffled -> flydsl",
            None,
            True,
            128,
            n,
            _is_flydsl,
        ),
        (
            "no tuned row + row-major B -> heuristic",
            None,
            False,
            128,
            n,
            _row_major_kid(128),
        ),
        (
            f"group 32: tuned kid{kid_pre32} + declared preshuffled -> runs it",
            _row(kid_pre32),
            True,
            32,
            n,
            lambda r: r == kid_pre32,
        ),
        (
            f"group 32: tuned kid{kid_pre} (group 128) + declared preshuffled -> g32 fallback",
            _row(kid_pre),
            True,
            32,
            n,
            _pre_fallback(32),
        ),
        (
            "group 32: no tuned row + declared preshuffled -> flydsl",
            None,
            True,
            32,
            n,
            _is_flydsl,
        ),
        (
            "group 32: no tuned row + row-major B -> g32 heuristic",
            None,
            False,
            32,
            n,
            _row_major_kid(32),
        ),
        (
            f"N=16, tuned kid{KID_PLAIN}, no preshuffled opus tile divides it -> raises",
            _row(KID_PLAIN),
            True,
            128,
            16,
            lambda r: isinstance(r, ValueError),
        ),
    )

    if not host_scale_kids:
        print("  skip host-rearranged-scale refusal: no such kid is built", flush=True)
    ok = True
    for label, row, b_preshuffled, group, n_, want in cases:
        got = _resolve(row, b_preshuffled, group, n_)
        good = want(got)
        ok &= good
        shown = (
            "ValueError"
            if isinstance(got, ValueError)
            else ("flydsl" if got == "flydsl" else f"kid{got}")
        )
        print(f"  {'ok  ' if good else 'FAIL'} {label}  [{shown}]", flush=True)
    return ok


def main():
    global KIDS_BPRESHUFFLE, KIDS_PLAIN_EXTRA
    p = argparse.ArgumentParser()
    p.add_argument("-g", type=int, default=2, help="batch (group) count")
    p.add_argument("-n", type=int, default=1024, help="N (multiple of 32)")
    p.add_argument(
        "-k",
        default="4096,2048,1024,512",
        help="comma-separated K list (multiples of 256); only 4096 has tuned "
        "rows, so the others run the public entry's fallback kid",
    )
    p.add_argument("--groups", default="128,32", help="comma-separated group sizes")
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
    ks = [int(x) for x in args.k.split(",") if x]
    groups = [int(x) for x in args.groups.split(",") if x]
    assert args.n % 32 == 0, "these kids tile N by 32"
    assert all(k % 256 == 0 for k in ks), "these kids tile K by 256"

    print("tables: one tuned CSV per B layout", flush=True)
    ok = _check_tables()
    print("dispatch: B-layout routing across the two public entries", flush=True)
    ok &= _check_dispatch()
    for group in groups:
        for k in ks:
            for m in [int(x) for x in args.sizes.split(",")]:
                ok &= _run(args.g, m, args.n, k, ydt, args.bench, args.split_k, group)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
