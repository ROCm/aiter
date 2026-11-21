# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf sweep for the fused MoE sorting kernels.

Two ops under test, both single-launch HIP kernels that also zero `moe_buf`:
  fused        fused_moe_sorting_fwd       sort only; caller runs its own router
  fused_topk   fused_moe_sorting_topk_fwd  router (sigmoid top-k) folded in

Every candidate (Opus, CK, fused, fused_topk) is compared exactly -- integer
equality on ids and block ids, bit equality on weights over the live slots --
against the torch reference shared with test_moe_sorting.py, and any mismatch
raises. The router inputs come from aiter's own grouped_topk so fused_topk is
judged against exactly what it replaces.

`main()` first runs the exactness suite (expert_mask, num_local_tokens / DP
padding, bf16 and fp32 gating, topk > 8, pad-slot contents), then the perf
sweep, whose rows are also checked exactly before timing.
"""

import argparse
import itertools
import os

import pandas as pd
import torch
from test_moe_sorting import _moe_sorting_roofline, run_torch_moe_sorting

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fused_moe_sorting import (
    fused_moe_sorting_fwd,
    fused_moe_sorting_is_supported,
    fused_moe_sorting_topk_fwd,
    fused_moe_sorting_topk_is_supported,
)
from aiter.ops.topk import biased_grouped_topk, grouped_topk
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

# Validated on MI355X only so far; the kernel is plain HIP with GFX9 DPP.
SUPPORTED_GFX = ["gfx950"]

# GLM-5.2 router: sigmoid scoring, one group, renormalised, scale 2.5.
ROUTER_N_GROUP = 1
ROUTER_TOPK_GROUP = 1
ROUTER_RENORM = True
ROUTER_SOFTMAX = False
ROUTER_SCALE = 2.5
SHARED_W = 1.0


def _alloc_outputs(token, topk, E, unit_size, model_dim, dtype):
    # Poisoned, so a slot the kernel fails to write cannot pass by accident.
    max_padded = int(token * topk + E * unit_size - topk)
    max_blocks = int((max_padded + unit_size - 1) // unit_size)
    max_padded = max_blocks * unit_size
    return {
        "sorted_ids": torch.full((max_padded,), -12345, dtype=dtypes.i32),
        "sorted_weights": torch.full((max_padded,), float("nan"), dtype=dtypes.fp32),
        "sorted_expert_ids": torch.full((max_blocks,), -12345, dtype=dtypes.i32),
        "num_valid_ids": torch.full((2,), -1, dtype=dtypes.i32),
        "moe_buf": torch.full((token, model_dim), 7.0, dtype=dtype),
    }


def _as_tuple(o):
    return (
        o["sorted_ids"],
        o["sorted_weights"],
        o["sorted_expert_ids"],
        o["num_valid_ids"],
        o["moe_buf"],
    )


def _check_exact(ref, out, topk, capacity, check_pad_weights=False, nan_equal=False):
    """Names of the outputs that differ from the torch reference (empty = exact).

    Compares the live range [0, num_valid_ids[0]) of the ids, the weights at the
    real slots (bit equality; a NaN never compares equal), the block ids, both
    counters and the zeroed moe_buf. With check_pad_weights the padding weights
    must be 0.0f as well, which is the Opus contract the fused kernels reproduce.
    With nan_equal a NaN weight matches a NaN in the same slot (a router row
    that is all -inf renormalises 0/0 in grouped_topk and in the kernel alike).
    """
    ref_ids, ref_w, ref_eids, ref_nvalid = ref
    ids, w, eids, nvalid, moe_buf = out
    capacity_errors = [
        name
        for name, actual, expected in (
            ("sorted_ids(capacity)", ids, ref_ids),
            ("sorted_weights(capacity)", w, ref_w),
            ("sorted_expert_ids(capacity)", eids, ref_eids),
        )
        if actual.shape != expected.shape
    ]
    if capacity_errors:
        return capacity_errors
    if not torch.equal(ref_nvalid, nvalid):
        return ["num_valid_ids"]
    bad = []
    n = int(ref_nvalid[0].item())
    sentinel = (topk << 24) | capacity
    live = ref_ids[:n] != sentinel
    if not torch.equal(ref_ids[:n], ids[:n]):
        bad.append("sorted_ids")
    rw, ow = ref_w[:n][live], w[:n][live]
    if nan_equal:
        same = torch.equal(rw.isnan(), ow.isnan()) and torch.equal(
            rw[~rw.isnan()], ow[~rw.isnan()]
        )
    else:
        same = torch.equal(rw, ow)
    if not same:
        bad.append("sorted_weights")
    if check_pad_weights and not torch.equal(
        w[:n][~live], torch.zeros_like(w[:n][~live])
    ):
        bad.append("sorted_weights(pad)")
    nblk = int(torch.count_nonzero(ref_eids != -1).item())
    if not torch.equal(ref_eids[:nblk], eids[:nblk]):
        bad.append("sorted_expert_ids")
    if moe_buf.numel() and torch.count_nonzero(moe_buf).item() != 0:
        bad.append("moe_buf")
    return bad


def _router_inputs(token, router_e, topk, gating_dtype, gating=None):
    if gating is None:
        gating = torch.randn((token, router_e), dtype=dtypes.fp32) * 3.0
    gating = gating.to(gating_dtype).contiguous()
    topk_weights = torch.empty((token, topk), dtype=dtypes.fp32)
    topk_ids = torch.empty((token, topk), dtype=dtypes.i32)
    grouped_topk(
        gating,
        topk_weights,
        topk_ids,
        ROUTER_N_GROUP,
        ROUTER_TOPK_GROUP,
        ROUTER_RENORM,
        ROUTER_SOFTMAX,
        ROUTER_SCALE,
    )
    return gating, topk_ids, topk_weights


def _run_fused(topk_ids, topk_weights, E, unit_size, model_dim, expert_mask, nlt):
    capacity, topk = topk_ids.shape
    o = _alloc_outputs(capacity, topk, E, unit_size, model_dim, dtypes.bf16)
    fused_moe_sorting_fwd(
        topk_ids,
        topk_weights,
        o["sorted_ids"],
        o["sorted_weights"],
        o["sorted_expert_ids"],
        o["num_valid_ids"],
        o["moe_buf"],
        E,
        int(unit_size),
        expert_mask,
        nlt,
    )
    return _as_tuple(o)


def test_fused_moe_sorting_exactness(model_dim=6144):
    """Branches the perf sweep does not reach; every case raises on mismatch."""
    torch.manual_seed(0)
    failures = []

    def expect(name, ref, out, topk, capacity, nan_equal=False):
        bad = _check_exact(
            ref, out, topk, capacity, check_pad_weights=True, nan_equal=nan_equal
        )
        if bad:
            failures.append(f"{name}: {', '.join(bad)}")

    E, topk, unit = 257, 8, 32

    # 1. expert_mask (EP): random local subset, all-local, one-local, none-local.
    for M in (1, 17, 64):
        _, ids, w = _router_inputs(M, E - 1, topk, dtypes.fp32)
        single = torch.zeros(E, dtype=dtypes.i32)
        single[int(ids[0, 0].item())] = 1
        masks = {
            "random": (torch.rand(E) < 0.5).to(dtypes.i32),
            "all": torch.ones(E, dtype=dtypes.i32),
            "none": torch.zeros(E, dtype=dtypes.i32),
            "single": single,
        }
        for tag, mask in masks.items():
            ref = run_torch_moe_sorting(ids, w, E, unit, mask, None)
            out = _run_fused(ids, w, E, unit, model_dim, mask, None)
            expect(f"expert_mask[{tag}] M={M}", ref, out, topk, M)

    # 2. num_local_tokens (DP-attention padding): static capacity 64, the live
    #    count is only ever read on the device. 999 must clamp to the capacity.
    cap = 64
    _, ids, w = _router_inputs(cap, E - 1, topk, dtypes.fp32)
    for live in (0, 1, 17, 63, 64, 999):
        nlt = torch.tensor([live], dtype=dtypes.i32)
        ref = run_torch_moe_sorting(ids, w, E, unit, None, nlt)
        out = _run_fused(ids, w, E, unit, model_dim, None, nlt)
        expect(f"num_local_tokens={live} cap={cap}", ref, out, topk, cap)

    # 3. topk > 8 on the sort-only path (vLLM appends the fused shared expert).
    for topk_n in (9, 16):
        for M in (8, 64):
            if not fused_moe_sorting_is_supported(M, E, topk_n, unit):
                continue
            ids = torch.stack([torch.randperm(E)[:topk_n] for _ in range(M)]).to(
                dtypes.i32
            )
            w = torch.rand((M, topk_n), dtype=dtypes.fp32)
            ref = run_torch_moe_sorting(ids, w, E, unit, None, None)
            out = _run_fused(ids, w, E, unit, model_dim, None, None)
            expect(f"topk={topk_n} M={M}", ref, out, topk_n, M)

    def run_fused_topk(
        tag, gating, ids, w, E_r, unit_size, nan_equal=False, bias=None, n_shared=0
    ):
        ref = run_torch_moe_sorting(ids, w, E_r, unit_size, None, None)
        M_r, topk_r = ids.shape
        o = _alloc_outputs(M_r, topk_r, E_r, unit_size, model_dim, dtypes.bf16)
        sync = torch.zeros(2, dtype=dtypes.i32)
        fused_moe_sorting_topk_fwd(
            gating,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E_r,
            topk_r,
            int(unit_size),
            ROUTER_RENORM,
            ROUTER_SOFTMAX,
            ROUTER_SCALE,
            None,
            None,
            sync,
            bias,
            n_shared,
            SHARED_W,
        )
        expect(tag, ref, _as_tuple(o), topk_r, M_r, nan_equal=nan_equal)
        if int(sync[0].item()) != 0:
            failures.append(f"{tag}: sync word not reset to 0")

    # 4. Fused router, fp32 and bf16 gating, against grouped_topk + torch sort.
    #    bf16 is the load-bearing case: duplicate scores inside the top-8 are
    #    common there and the arg-max tie order has to match aiter's exactly.
    router_e = E - 1
    for gating_dtype in (dtypes.fp32, dtypes.bf16):
        for M in (1, 16, 64, 80):
            for unit_size in (16, 32):
                if not fused_moe_sorting_topk_is_supported(
                    M, E, topk, unit_size, router_e
                ):
                    continue
                gating, ids, w = _router_inputs(M, router_e, topk, gating_dtype)
                tag = f"fused_topk gating={gating_dtype} M={M} unit={unit_size}"
                run_fused_topk(tag, gating, ids, w, E, unit_size)

    # 5. -inf logits: sigmoid(-inf) = 0 is a live score in grouped_topk, so a
    #    row with fewer than topk finite logits fills its top-k from the 0s
    #    (tie order included), and an all -inf row picks k zero-score experts
    #    with NaN weights after renorm. Router widths cover no padding (256), and
    #    padding vectors in the NREG 4 (160) and NREG 8 (384) register routers,
    #    which must never win even when every real score is 0; each one also
    #    runs on the LDS router.
    for router_e, M in ((256, 64), (160, 64), (384, 32)):
        E_inf = router_e + 1
        g = torch.randn((M, router_e), dtype=dtypes.fp32) * 3.0
        for t in range(M):
            n_fin = t % (topk + 2)  # 0..topk+1 finite logits, rest -inf
            keep = torch.randperm(router_e)[:n_fin]
            row = torch.full((router_e,), float("-inf"))
            row[keep] = g[t, keep]
            g[t] = row
        for gating_dtype in (dtypes.fp32, dtypes.bf16):
            for lds in (False, True):
                if not fused_moe_sorting_topk_is_supported(
                    M, E_inf, topk, unit, router_e
                ):
                    continue
                gating, ids, w = _router_inputs(M, router_e, topk, gating_dtype, g)
                tag = (
                    f"-inf rows router_e={router_e} gating={gating_dtype} M={M}"
                    f" router={'lds' if lds else 'reg'}"
                )
                if lds:
                    os.environ["AITER_FUSED_MOE_SORTING_ROUTER_LDS"] = "1"
                try:
                    run_fused_topk(tag, gating, ids, w, E_inf, unit, nan_equal=True)
                finally:
                    os.environ.pop("AITER_FUSED_MOE_SORTING_ROUTER_LDS", None)

    # 6. Biased routing + a fused shared expert -- the GLM-5.2 / DeepSeek decode
    #    preamble as vLLM drives it: experts chosen on sigmoid + bias, weighted by
    #    the unbiased sigmoid, plus one shared slot (id router_e, fixed weight)
    #    that renorm and the scale never touch. The reference is aiter's own
    #    biased_grouped_topk writing the routed columns of a [M, topk + 1] buffer
    #    whose shared column is preset, which is exactly what vLLM hands aiter.
    #    The bias is randn: its ties have to resolve like the stock kernel's.
    for router_e, M_list in ((256, (1, 16, 64, 80)), (384, (1, 32))):
        E_b = router_e + 1
        bias32 = torch.randn(router_e, dtype=dtypes.fp32)
        for gating_dtype in (dtypes.fp32, dtypes.bf16):
            for M in M_list:
                for n_shared in (0, 1):
                    topk_t = topk + n_shared
                    if not fused_moe_sorting_topk_is_supported(
                        M, E_b, topk_t, unit, router_e, n_shared
                    ):
                        continue
                    torch.manual_seed(100 + M)
                    gating = (
                        (torch.randn((M, router_e), dtype=dtypes.fp32) * 3.0)
                        .to(gating_dtype)
                        .contiguous()
                    )
                    bias = bias32.to(gating_dtype)  # stock reads it in gating dtype
                    ids = torch.empty((M, topk_t), dtype=dtypes.i32)
                    w = torch.empty((M, topk_t), dtype=dtypes.fp32)
                    ids[:, topk:] = router_e
                    w[:, topk:] = SHARED_W
                    biased_grouped_topk(
                        gating,
                        bias,
                        w[:, :topk],
                        ids[:, :topk],
                        1,
                        1,
                        ROUTER_RENORM,
                        ROUTER_SCALE,
                    )
                    for lds in (False, True):
                        tag = (
                            f"biased router_e={router_e} gating={gating_dtype} M={M}"
                            f" shared={n_shared} router={'lds' if lds else 'reg'}"
                        )
                        if lds:
                            os.environ["AITER_FUSED_MOE_SORTING_ROUTER_LDS"] = "1"
                        try:
                            run_fused_topk(
                                tag,
                                gating,
                                ids,
                                w,
                                E_b,
                                unit,
                                bias=bias,
                                n_shared=n_shared,
                            )
                        finally:
                            os.environ.pop("AITER_FUSED_MOE_SORTING_ROUTER_LDS", None)

    if failures:
        raise AssertionError(
            "fused_moe_sorting exactness failures:\n  " + "\n  ".join(failures)
        )
    aiter.logger.info("fused_moe_sorting exactness suite: all cases exact")


@benchmark()
def test_fused_moe_sorting(token, E, topk, unit_size, model_dim, dtype):
    # GLM-5.2 routes over n_routed_experts=256 while the sort sees E=257 (the
    # shared expert is not routed), so the gating tile is [token, E-1] there.
    router_e = E - 1 if E == 257 else E
    gating, topk_ids, topk_weights = _router_inputs(token, router_e, topk, dtypes.fp32)

    ref = run_torch_moe_sorting(topk_ids, topk_weights, E, unit_size, None, None)

    ws_size = aiter.moe_sorting_opus_get_workspace_size(token, E, topk, 0)
    opus_ws = torch.empty(ws_size, dtype=torch.uint8) if ws_size > 0 else None
    # Cross-block handshake word for the fused router; zero once, kernel
    # leaves it zero.
    sync = torch.zeros(2, dtype=dtypes.i32)

    outs = {}

    def opus():
        o = outs["opus"]
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
            opus_ws,
            0,
            None,
            None,
            None,
        )
        return _as_tuple(o)

    def ck():
        o = outs["ck"]
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
            0,
        )
        return _as_tuple(o)

    def fused():
        o = outs["fused"]
        fused_moe_sorting_fwd(
            topk_ids,
            topk_weights,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            int(unit_size),
            None,
            None,
        )
        return _as_tuple(o)

    def fused_topk():
        o = outs["fused_topk"]
        fused_moe_sorting_topk_fwd(
            gating,
            o["sorted_ids"],
            o["sorted_weights"],
            o["sorted_expert_ids"],
            o["num_valid_ids"],
            o["moe_buf"],
            E,
            topk,
            int(unit_size),
            ROUTER_RENORM,
            ROUTER_SOFTMAX,
            ROUTER_SCALE,
            None,
            None,
            sync,
        )
        return _as_tuple(o)

    candidates = {"opus": opus, "ck": ck}
    if fused_moe_sorting_is_supported(token, E, topk, int(unit_size)):
        candidates["fused"] = fused
    if fused_moe_sorting_topk_is_supported(token, E, topk, int(unit_size), router_e):
        candidates["fused_topk"] = fused_topk

    flops, nbytes = _moe_sorting_roofline(token, topk, E, model_dim, dtype)
    ret = {"gfx": get_gfx()}
    failures = {}
    for name, fn in candidates.items():
        outs[name] = _alloc_outputs(token, topk, E, unit_size, model_dim, dtype)
        # Correctness on a poisoned buffer first, then timing. Pad-slot weights
        # are part of the contract for Opus and the fused kernels; CK is not
        # held to it.
        bad = _check_exact(ref, fn(), topk, token, check_pad_weights=name != "ck")
        if bad:
            failures[name] = bad
        _, us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} mismatch"] = len(bad)
    if failures:
        raise AssertionError(
            f"moe sorting mismatch vs torch reference at M={token} E={E} "
            f"topk={topk} unit_size={unit_size}: {failures}"
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fused_moe_sorting unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        help="moe_buf data type.\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 8, 16, 32, 64, 80],
        help="Number of tokens.\n    e.g.: -m 64",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[257, 256, 128],
        help="Number of experts (paired with -t).\n    e.g.: -e 257",
    )
    parser.add_argument(
        "-t",
        "--topk",
        type=int,
        nargs="*",
        default=[8, 8, 8],
        help="Top-k per token (paired with -e).\n    e.g.: -t 8",
    )
    parser.add_argument(
        "-u",
        "--unit_size",
        type=int,
        nargs="*",
        default=[16, 32],
        help="Rows per expert block.\n    e.g.: -u 32",
    )
    parser.add_argument(
        "-md",
        "--model_dim",
        type=int,
        nargs="*",
        default=[6144],
        help="Model dimension (moe_buf width).\n    e.g.: -md 6144",
    )
    parser.add_argument(
        "--skip-exactness",
        action="store_true",
        help="Skip the exactness suite and only run the perf sweep.",
    )
    args = parser.parse_args()
    assert len(args.expert) == len(args.topk), "-e and -t must pair up"

    if not args.skip_exactness:
        test_fused_moe_sorting_exactness(model_dim=args.model_dim[0])

    df = []
    for dtype, m, (E, topk), unit_size, model_dim in itertools.product(
        args.dtype, args.m, zip(args.expert, args.topk), args.unit_size, args.model_dim
    ):
        df.append(test_fused_moe_sorting(m, E, topk, unit_size, model_dim, dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df.to_markdown(index=False)}")


if __name__ == "__main__":
    main()
