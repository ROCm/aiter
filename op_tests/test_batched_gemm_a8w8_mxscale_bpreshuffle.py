# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf check for the fp8 e8m0 mxscale BMM B-preshuffle path.

Named after the public entry ``batched_gemm_a8w8_mxscale_bpreshuffle`` rather
than after opus, because that entry now dispatches to either the opus kids or
the FlyDSL backend depending on the shape, and both are exercised here.

Kids are grouped by their ``GROUP_N x GROUP_K`` block. Each block gets its own
weight, w_scale and reference: 32x32 and 128x128 scale a square N/K block, 1x32
scales every N row on its own, and only the matching w_scale shape is accepted.

kid 170 is kid 320's tile (64x32x256 -> a 32x32x256 per-wave register tile, i.e.
COM_REP_M/N/K = 2/2/2) with two compile-time axes flipped: the weight comes from
``shuffle_weight(w, layout=(16, 16))`` and every MFMA picks its e8m0 byte with
the hardware ``scale_op_sel`` immediate instead of a broadcast pack. kid 171 adds
a third: since the preshuffle order already is the mfma_16x16x128 B fragment
order, its consumer waves ``buffer_load`` B straight into the MFMA registers and
B never touches LDS.

Within a block all kids run on the same quantized data -- kid 320 on the
row-major weight, 170 and 171 on the preshuffled one -- so a mismatch localizes
to the preshuffle / op_sel / direct-B changes rather than to the quantization.
Every kid is scored against the torch reference, not against another kid.

The compact regression covers runtime dimensions, short K and segment tails.

Usage:
    python3 op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py
    python3 -m pytest -q op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py
    python3 op_tests/test_batched_gemm_a8w8_mxscale_bpreshuffle.py -g 4 -n 1024 -k 4096
"""

import argparse
import os
import sys

# Direct execution adds op_tests/ to sys.path; shared helpers need its parent.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.opus import opus_bmm, policy
from aiter.ops.opus.gemm_op_a8w8 import (
    _mxscale_bmm_workspace,
    _opus_gemm_a8w8_mxscale_bmm_launch_raw,
    bmm_a8w8_mxscale_opus_bpreshuffle,
)
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import run_perftest
from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_bmm_bpreshuffle_compact_kernels_list as _COMPACT_KERNELS,
)
from op_tests.test_opus_a8w8_bmm import (
    GROUP,
    _block_varied,
    _preshuffled_kids,
    _quant_block_e8m0,
    _quant_per_token_e8m0,
    _scale_picker,
    run_torch,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX,
    reason="gfx950 required",
)
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

# Measured against the dequantized torch reference. The e8m0 quant floor on these
# shapes is ~0.0014 and every kid, row-major and preshuffled alike, sits exactly
# on it, so this leaves about 2x headroom -- the same gate the tuner and
# op_tests/test_opus_a8w8_bmm.py use.
_ERR_TOL = 0.003


def _rel_err(y, ref):
    return (y.float() - ref).abs().mean().item() / (ref.abs().mean().item() + 1e-9)


def _run(g, m, n, k, ydt, bench, split_k=1, group=GROUP):
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16, group=group)

    O_in = O_mx.transpose(0, 1)  # [m, g, k] mmajor view
    xs_in = xs_mx.transpose(0, 1)  # [m, g, k/group] view

    # One weight, scale and reference per GROUP_N. The 1x32 kids share GROUP_K
    # with the square ones but scale per N row, so their w_scale is [G,N,K/32]
    # rather than [G,N/32,K/32] -- handing them the square one is not a stricter
    # test, it is a shape their launcher refuses. Quantized on demand so a run
    # that selects no kid of a given GROUP_N does not pay for it.
    _data = {}

    def data_for(group_n):
        if group_n not in _data:
            W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(
                W_bf16, group=group, group_n=group_n
            )
            _data[group_n] = (
                W_mx,
                # Same bytes, 16x16-tiled: [G,N,K] -> [G][N/16][K/32][2][16n][16k].
                shuffle_weight(W_mx, layout=(16, 16)),
                ws_mx,
                _scale_picker(xs_mx, ws_mx, n, k),
                run_torch(
                    O_mx, W_mx, xs_fp32, ws_fp32, group=group, group_n=group_n
                ).transpose(
                    0, 1
                ),  # [m, g, n]
            )
        return _data[group_n]

    def _call(kid, W, scale_for):
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
    kid_group_n = {}  # kid -> the GROUP_N whose data it ran on, for the bench pass
    skipped = []
    kid_plain = KID_PLAIN + (1000 if group == 32 else 0)
    for kid in KIDS_BPRESHUFFLE:
        # Split the whole block, not just GROUP_K: GROUP_N picks which weight.
        group_n, group_k = (
            int(v) for v in policy.mxscale_bmm_kid_block(kid).split("x")
        )
        if group_k != group:
            continue
        _, W_sh, _, scale_for, ref = data_for(group_n)
        try:
            errs[kid] = _rel_err(_call(kid, W_sh, scale_for), ref)
            kid_group_n[kid] = group_n
        except (RuntimeError, ValueError):
            skipped.append(kid)
    # Every block this GROUP_K has in the catalog must have been attempted. The
    # bug this catches is not a wrong number, it is a missing row: a selection
    # rule that drops a block sends all of its kids into `skipped`, which reads
    # as "this tile does not divide this shape", and the run still prints PASS
    # with twelve fewer entries than nobody counts. Asserted on attempted, not
    # on scored, because a shape where every kid of a block legitimately skips
    # is allowed.
    want = {
        policy.mxscale_bmm_kid_block(kid)
        for kid in KIDS_BPRESHUFFLE
        if policy.mxscale_bmm_kid_block(kid).endswith(f"x{group}")
    }
    got = {f"{group_n}x{group}" for group_n in _data}
    assert got == want, f"GROUP_K={group} blocks not attempted: {sorted(want - got)}"

    # The row-major reference kid, or at a K too short for it the 128-deep one.
    # Square only -- every 1x32 kid is preshuffled, so that block has no
    # row-major member to run here.
    #
    # Both candidates can be out of reach at once: the K-tile floor these kids
    # carry is a function of K *and* split_k, so a short K at split_k > 1 leaves
    # neither one dividing the shape (K=1024 split_k=2 is the case that used to
    # take the whole sweep down). That is the same "absent from the row" as any
    # skipped kid -- this one sets no tolerance, it is a reading aid, see the
    # gate below -- so report it n/a and carry on. The refusal arrives as
    # ValueError from the launch planner or RuntimeError from the kernel
    # depending on which layer catches it, which is why both are caught here,
    # matching the kid loop above.
    W_mx, _, _, scale_for_sq, ref_sq = data_for(group)
    err_plain = None
    plain_exc = None
    for cand in (kid_plain, kid_plain - KID_PLAIN + 8653):
        try:
            err_plain = _rel_err(_call(cand, W_mx, scale_for_sq), ref_sq)
        except (RuntimeError, ValueError) as e:
            plain_exc = e
            continue
        kid_plain = cand
        break

    # The public entry end to end: guarded custom op -> the preshuffle table
    # (the entry picks that one, nothing to set) -> the row's backend, opus or
    # flydsl, or flydsl's heuristic for a shape the table lacks.
    # It infers the block from the w_scale shape, so each GROUP_N is a separate
    # route through it. Only those with a kid that ran: where every kid of a
    # GROUP_N was skipped, the shape has no kernel and the entry has nothing to
    # reach, which is the skip, not a failure.
    err_pub = {}
    if split_k == 1:  # the entry defaults splitK, so only compare where they agree
        for group_n in sorted(set(kid_group_n.values())):
            _, W_sh, ws_mx, _, ref = _data[group_n]
            err_pub[group_n] = _rel_err(
                batched_gemm_a8w8_mxscale_bpreshuffle(
                    O_in.contiguous(), W_sh, xs_in.contiguous(), ws_mx, dtype=ydt
                ),
                ref,
            )

    row = f"G{group:<3} g={g:<3} m={m:<6} n={n:<6} k={k:<6} sk={split_k} "
    row += "  ".join(f"kid{kid}={e:.5f}" for kid, e in errs.items())
    # Named even when absent: a silently missing reference reads as "nobody ran
    # it", which is exactly how a real regression would look.
    row += (
        f"  kid{kid_plain}={err_plain:.5f}"
        if err_plain is not None
        else f"  kid{kid_plain}=n/a({type(plain_exc).__name__})"
    )
    row += "  public=" + (
        "off"
        if not err_pub
        else " ".join(f"{gn}x{group}:{e:.5f}" for gn, e in err_pub.items())
    )
    if skipped:
        row += "  skipped=" + ",".join(str(k) for k in skipped)
    if bench:

        def _time(kid, W, scale_for):
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

        times = [
            f"kid{kid}={_time(kid, _data[kid_group_n[kid]][1], _data[kid_group_n[kid]][3]):.2f}us"
            for kid in errs
        ]
        # Same refusals as the correctness pass: a kid whose tile does not
        # divide this shape is absent from the timings, not a crash. Timing a
        # kid the pass above could not score would mean running it anyway.
        if err_plain is not None:
            times.append(f"kid{kid_plain}={_time(kid_plain, W_mx, scale_for_sq):.2f}us")
        for kid in KIDS_PLAIN_EXTRA:
            try:
                times.append(f"kid{kid}={_time(kid, W_mx, scale_for_sq):.2f}us")
            except (RuntimeError, ValueError):
                pass
        row += "  |  " + "  ".join(times)
    print(row, flush=True)
    # Every kid is judged against the torch reference, not against the row-major
    # kid. kid_plain stays in the row as a reading aid -- it is the same tile with
    # B row-major, so a preshuffle kid drifting away from it is still the first
    # thing to look at -- but it no longer sets the bar, because anchoring the
    # tolerance to another kernel makes the gate move whenever that kernel does.
    ok = all(e <= _ERR_TOL for e in errs.values())
    return ok and all(e <= _ERR_TOL for e in err_pub.values())


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


# Fast path, partial M/N tiles, and short/multi-tile K segment tails.
COMPACT_SHAPES = [
    (3, 7, 128, 128),
    (2, 17, 256, 512),
    (3, 97, 384, 768),
    (3, 95, 256, 8192),
    (2, 193, 512, 8448),
    (16, 2047, 128, 256),
    (1, 1, 1024, 4096),
    (3, 65, 384, 4224),
    (2, 129, 512, 4608),
]


def _compact_inputs(b, m, n, k, dtype, seed=41):
    torch.manual_seed(seed)

    def varied(shape):
        values = torch.randn(shape, device="cuda", dtype=torch.float32)
        amplitude = torch.exp2(
            torch.randint(-3, 4, (*shape[:-1], k // 128, 1), device="cuda").float()
        )
        return (
            (values.view(*shape[:-1], k // 128, 128) * amplitude).view(shape).bfloat16()
        )

    x, xs, xs_float = _quant_per_token_e8m0(varied((b, m, k)))
    w, ws, ws_float = _quant_block_e8m0(varied((b, n, k)))
    ref = run_torch(x, w, xs_float, ws_float).transpose(0, 1).to(dtype)
    return (
        x.transpose(0, 1).contiguous(),
        shuffle_weight(w, (16, 16)),
        xs.transpose(0, 1).contiguous(),
        ws,
        ref,
    )


def _check_compact_result(y, ref, storage):
    assert torch.isfinite(y).all()
    delta = (y.float() - ref.float()).abs()
    assert (delta.mean() / ref.float().abs().mean().clamp_min(1e-9)).item() < 0.001
    assert (delta > 0.01 + 0.01 * ref.float().abs()).float().mean().item() < 0.001
    assert (storage[:64] == 123).all() and (storage[-64:] == 123).all()


@pytest.mark.parametrize("shape", COMPACT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compact_mxscale_bmm(shape, dtype):
    b, m, n, k = shape
    x, w, xs, ws, ref = _compact_inputs(*shape, dtype)
    storage = torch.full((m * b * n + 128,), 123, dtype=dtype, device="cuda")
    y = storage[64:-64].view(m, b, n)
    kids = [kid for kid in KIDS_BPRESHUFFLE if kid in _COMPACT_KERNELS]
    assert kids, "No compact kernels selected"
    for kid in kids:
        if k % _COMPACT_KERNELS[kid].B_K:
            continue
        y.fill_(float("nan"))
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(
            x, w, y, xs, ws, workspace=None, kid=kid, split_k=1
        )
        torch.cuda.synchronize()
        _check_compact_result(y, ref, storage)
        first = y.clone()
        y.fill_(float("nan"))
        opus_bmm(
            x.transpose(0, 1),
            w,
            y.transpose(0, 1),
            kid=kid,
            layout="mxscale_bmm",
            x_scale=xs.transpose(0, 1),
            w_scale=ws,
        )
        assert torch.equal(y, first)


# --- GROUP_N == 1, GROUP_K == 32 -----------------------------------------
# These are bpreshuffle-only: there is no row-major 1x32 kid, so `_run` above
# cannot host them -- its gate is "match the row-major twin's accuracy" and
# there is no twin. ``check_1x32_kids`` in op_tests/test_opus_a8w8_bmm.py
# checks the kernels against a dequantized reference with the kid pinned; what
# is left, and what lives here, is the public entry: route selection, the
# per-row scale reaching the kernel intact, and capture.
_1X32_SHAPE = (2, 129, 512, 4608)  # B, M, N, K -- M and K deliberately untidy


def _inputs_1x32(b, m, n, k, dtype=torch.bfloat16, seed=197):
    """Per-N-row K32 e8m0 scales, plus a dequantized fp32 reference."""
    torch.manual_seed(seed)

    def quant(shape):
        # Vary every row and K32 block independently: a square grouping or a
        # wrong MFMA lane's scale then moves the result by a power of two,
        # which a uniform magnitude would absorb.
        blocks = torch.randn((*shape[:-1], k // 32, 32), device="cuda")
        exponents = (
            torch.arange(blocks.numel() // 32, device="cuda").reshape(
                *shape[:-1], k // 32, 1
            )
            % 9
            - 4
        )
        values = (blocks * torch.exp2(exponents.float())).reshape(shape).bfloat16()
        q, scales = per_1x32_f8_scale_f8_quant(values, scale_type=torch.uint8)
        scales = scales.view(*shape[:-1], k // 32)
        dequant = q.float() * torch.exp2(
            scales.view(torch.uint8).float() - 127
        ).repeat_interleave(32, dim=-1)
        return q, scales, dequant

    x, xs, xd = quant((m, b, k))
    w, ws, wd = quant((b, n, k))
    ref = torch.einsum("mbk,bnk->mbn", xd, wd).to(dtype)
    return x, shuffle_weight(w, (16, 16)), xs, ws, ref


def test_public_entry_1x32():
    """The public entry must route 1x32 scales to a 1x32 kid and stay exact.

    ``check_1x32_kids`` pins the kid, so it cannot see a routing regression:
    a 1x32 w_scale handed to a square kid is refused outright, and a square
    scale sent to a 1x32 kid is the mirror of that. This is the only check that
    lets the entry choose.
    """
    x, w, xs, ws, ref = _inputs_1x32(*_1X32_SHAPE)
    y = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws)
    torch.cuda.synchronize()
    # Aggregate, not elementwise. At K=4608 of signed data the accumulation
    # cancels, so outputs near zero sit further from a dequantized reference
    # than any elementwise tolerance worth setting, while the aggregate stays
    # orders tighter than the gate. The exactness this test needs is checked
    # below, against the kernel itself rather than against the reference.
    rel = (y.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    assert rel.item() < 0.003, f"public entry rel err {rel.item():.5f}"

    # A tuned row naming a square-block kid must fall back to the matching 1x32
    # kid rather than be taken at face value -- this is the drift a retune
    # leaves behind when the scale layout changes under a stale CSV. Bit-exact
    # against the entry's own pick: a fallback landing anywhere else shows up
    # here even when both answers are numerically acceptable.
    stale_square_kid = next(
        kid for kid in KIDS_BPRESHUFFLE if policy.mxscale_bmm_kid_block(kid) == "32x32"
    )
    fallback = bmm_a8w8_mxscale_opus_bpreshuffle(
        x,
        w,
        xs,
        ws,
        kernelId=stale_square_kid,
        group_n=1,
        group_size=32,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(fallback, y, rtol=0, atol=0)


def test_rejects_square_scale_shape_on_1x32_kid():
    """A square w_scale on a 1x32 kid must be refused, not quietly misread."""
    x, w, xs, ws, ref = _inputs_1x32(*_1X32_SHAPE)
    kid = next(k for k in KIDS_BPRESHUFFLE if policy.mxscale_bmm_kid_block(k) == "1x32")
    square_ws = ws[:, ::32, :].contiguous()  # [B, N/32, K/32] -- the wrong block
    with pytest.raises((ValueError, RuntimeError), match="w_scale"):
        opus_bmm(
            x.transpose(0, 1),
            w,
            torch.empty_like(ref).transpose(0, 1),
            kid=kid,
            layout="mxscale_bmm",
            x_scale=xs.transpose(0, 1),
            w_scale=square_ws,
        )


def test_cuda_graph_replay_1x32():
    """Capture + replay on the 1x32 route, which has its own scale plumbing."""
    x, w, xs, ws, _ = _inputs_1x32(*_1X32_SHAPE)
    eager = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws)
    for _ in range(2):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(captured).all(), "replay left the buffer untouched"
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_cuda_graph_replay():
    """The public entry must be capturable and replay bit-identically.

    Nothing else here runs this path under a graph, and graph capture is where
    a kernel that allocates, synchronizes, or reads a host value at launch time
    stops working -- none of which the eager calls above can see. Serving
    frameworks capture this entry, so a regression would surface there first.

    The buffer is refilled with NaN before each replay: a graph that captured
    nothing, or whose output pointer drifted, leaves the NaNs behind rather
    than quietly re-presenting the previous result.
    """
    # bf16 only: this shape routes to the FlyDSL backend, whose BMM declines a
    # fp32 output. Pinning the dtype keeps the test about capture rather than
    # about which backend a shape happens to land on.
    dtype = torch.bfloat16
    b, m, n, k = 2, 129, 512, 4608
    x, w, xs, ws, _ = _compact_inputs(b, m, n, k, dtype)
    eager = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws, dtype=dtype)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = batched_gemm_a8w8_mxscale_bpreshuffle(x, w, xs, ws, dtype=dtype)
    for _ in range(2):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(captured).all(), "replay left the buffer untouched"
        # Same inputs and same kid, so this is determinism, not tolerance.
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def _splitk_kid(x, w, xs, ws, ref, split_k):
    """First bpreshuffle kid that runs this shape at ``split_k``, or None.

    Probed rather than hard-coded: which tiles divide a shape at a given split
    is a property of the catalog, so a pinned kid would turn a retuning into a
    failure here instead of a skip.
    """
    for kid in KIDS_BPRESHUFFLE:
        y = torch.zeros_like(ref)
        try:
            _opus_gemm_a8w8_mxscale_bmm_launch_raw(
                x,
                w,
                y,
                xs,
                ws,
                workspace=_mxscale_bmm_workspace(x, w, y, kid, split_k),
                kid=kid,
                split_k=split_k,
            )
        except (RuntimeError, ValueError):
            continue
        torch.cuda.synchronize()
        if _rel_err(y, ref) <= _ERR_TOL:
            return kid
    return None


def test_split_k_arrival_counters():
    """The fused split-K arrival counters, which nothing else here reaches.

    splitK > 1 is the only path that takes them, and the buffer is cached per
    (device, stream) and reused for the life of the process, so its whole
    contract is about what the *next* launch sees. Three things are checked,
    cheapest first, because the cheap one fails earliest and reads clearest:

    Repeat: the last split re-arms each counter, so a second launch on the
    same buffer must agree with the first. Drop the re-arm and the second
    launch finds every counter already at split_k, nobody recognises itself
    as last, and Y is simply never written -- which is what this catches.

    Capture: capture runs on a side stream, which owns no buffer unless it was
    warmed, and allocating mid-capture is illegal. Unwarmed, the launcher must
    fall back to the workspace + reducer path; warmed, the fused path itself
    is captured. Both must produce the eager numbers, so a broken fallback and
    an allocation escaping into capture both show up here.

    Replay: replay re-runs the kernel with the captured pointer and no host
    code in between, so a counter left dirty by replay N breaks replay N+1 --
    invisible on a single replay, hence two.

    Not parametrized on warm/unwarm: the counter buffer is process-wide static
    state, so a failure in one parametrization perturbs the kid probe in the
    next and the second run silently measures a different kernel.
    """
    split_k = 2
    b, m, n, k = 2, 129, 512, 4608  # K deep enough to survive the split
    x, w, xs, ws, ref = _compact_inputs(b, m, n, k, torch.bfloat16)
    kid = _splitk_kid(x, w, xs, ws, ref, split_k)
    if kid is None:
        pytest.skip(f"no bpreshuffle kid runs {b}x{m}x{n}x{k} at split_k={split_k}")

    y = torch.zeros_like(ref)
    # Allocated once and reused: a workspace allocated inside the capture would
    # land in the graph's private pool, making this a test of the pool rather
    # than of the counters.
    workspace = _mxscale_bmm_workspace(x, w, y, kid, split_k)

    def launch():
        _opus_gemm_a8w8_mxscale_bmm_launch_raw(
            x, w, y, xs, ws, workspace=workspace, kid=kid, split_k=split_k
        )

    def run_eager():
        y.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        assert torch.isfinite(y).all(), "no workgroup reduced the partials"
        return y.clone()

    # Two launches, not one: the first only proves the kernel works on a clean
    # buffer, which the probe above already did. The second is the one that
    # meets a counter the first launch had to re-arm.
    eager = run_eager()
    assert _rel_err(eager, ref) <= _ERR_TOL, "eager split-K is already wrong"
    torch.testing.assert_close(run_eager(), eager, rtol=0, atol=0)

    for warm_capture_stream in (False, True):
        side = torch.cuda.Stream()
        if warm_capture_stream:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                launch()
            torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            launch()
        for _ in range(2):
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(
                y
            ).all(), f"replay left the buffer untouched (warm={warm_capture_stream})"
            torch.testing.assert_close(y, eager, rtol=0, atol=0)


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
    if args.split_k == 1 and any(kid in _COMPACT_KERNELS for kid in KIDS_BPRESHUFFLE):
        print("compact: runtime dimensions and K segment tails", flush=True)
        for shape in COMPACT_SHAPES:
            test_compact_mxscale_bmm(shape, ydt)
        print("cuda graph: capture + replay of the public entry", flush=True)
        test_cuda_graph_replay()
        print("1x32: public entry route, square-scale refusal, capture", flush=True)
        test_public_entry_1x32()
        test_rejects_square_scale_shape_on_1x32_kid()
        test_cuda_graph_replay_1x32()
        # Guarded by the same split_k == 1 as the rest of this block because
        # it is a one-off structural check, not part of the sweep -- it pins
        # its own split_k = 2 regardless of the flag.
        print(
            "split-K: arrival counters across launches, capture and replay", flush=True
        )
        test_split_k_arrival_counters()
    for group in groups:
        for k in ks:
            for m in [int(x) for x in args.sizes.split(",")]:
                ok &= _run(args.g, m, args.n, k, ydt, args.bench, args.split_k, group)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
