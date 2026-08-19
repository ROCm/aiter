# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf tests for the FlyDSL warp-decode MoE kernels (SILOTIGER-667).

Two layers, both in this one file (per the ticket's locked testing standard, ?2):

  * **Correctness (pytest gate):** the three low-level primitives in isolation
    (``v_dot2_f32_bf16``, ``cvt_scalef32_pk_bf16_fp8``, 64-lane butterfly reduce)
    plus the ``gate_up`` / ``down_reduce`` FP8 fast paths vs a torch reference.
  * **Perf sweep (``__main__``):** ``@benchmark`` + ``run_perftest`` +
    ``checkAllclose`` over realistic decode shapes, emitting a markdown table with
    ``us`` / ``TFLOPS`` / ``TB/s`` / ``%peak`` / ``err`` per stage. Timing uses the
    shared harness (device time by default), adequate warmup/iters, and a
    cold-HBM-read rotation policy (see ``_rotate_for``). Never hand-rolled timers.

Usage:
    # correctness (fast):
    pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py
    # correctness + perf sweep with markdown tables:
    python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py
    python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py --timing cuda_event
"""

from __future__ import annotations

import argparse

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL warp-decode tests.",
        allow_module_level=True,
    )

import flydsl.compiler as flyc  # noqa: E402
import pandas as pd  # noqa: E402

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg  # noqa: E402
from aiter.ops.flydsl.kernels.warp_decode_moe import (  # noqa: E402
    WARP_SIZE,
    build_warp_decode_primitives_module,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest  # noqa: E402

torch.set_default_device("cuda")

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")

# Approximate gfx950 (MI355X, HBM3E) peak DRAM bandwidth, for the %peak column
# only. Perf figures are streamed-weight TB/s; %peak is illustrative, not a gate.
_HBM_PEAK_TBS = 8.0

# Perf-sweep default timing knobs (SILOTIGER-667 ?2: >=5 warmup, >=100 iters for
# these tiny B=1 decode kernels; small 2/1 is reserved for correctness-only).
_PERF_NUM_ITERS = 100
_PERF_NUM_WARMUP = 5


def _run_primitives(serialize_dot2: bool = True):
    """Launch the primitive kernel once and return (inputs, outputs) dicts."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260404)
    n = WARP_SIZE

    # 1. dot2 inputs: two bf16 lanes packed per i32.
    a_bf16 = (torch.rand((n, 2), generator=gen, device=device) * 4 - 2).to(
        torch.bfloat16
    )
    b_bf16 = (torch.rand((n, 2), generator=gen, device=device) * 4 - 2).to(
        torch.bfloat16
    )
    a_i32 = a_bf16.contiguous().view(torch.int32).reshape(n).contiguous()
    b_i32 = b_bf16.contiguous().view(torch.int32).reshape(n).contiguous()
    out_dot = torch.zeros(n, dtype=torch.float32, device=device)

    # 2. convert inputs: four e4m3 bytes packed per i32, one f32 scale per lane.
    f8 = (torch.rand((n, 4), generator=gen, device=device) * 8 - 4).to(
        torch.float8_e4m3fn
    )
    f8_i32 = f8.contiguous().view(torch.int32).reshape(n).contiguous()
    scale = torch.empty(n, dtype=torch.float32, device=device)
    # Power-of-two scales keep the reference exact regardless of whether the HW
    # applies the full f32 or only its exponent.
    scale[0::3] = 1.0
    scale[1::3] = 2.0
    scale[2::3] = 0.5
    out_cvt = torch.zeros(n * 4, dtype=torch.bfloat16, device=device)

    # 3. reduce inputs.
    red_in = (torch.rand(n, generator=gen, device=device) * 2 - 1).to(torch.float32)
    out_red = torch.zeros(n, dtype=torch.float32, device=device)

    launcher = build_warp_decode_primitives_module(serialize_dot2=serialize_dot2)
    cf = flyc.compile(
        launcher,
        ptr_arg(a_i32),
        ptr_arg(b_i32),
        ptr_arg(out_dot),
        ptr_arg(f8_i32),
        ptr_arg(scale),
        ptr_arg(out_cvt),
        ptr_arg(red_in),
        ptr_arg(out_red),
        torch.cuda.current_stream(),
    )
    del cf
    torch.cuda.synchronize()

    inputs = {
        "a_bf16": a_bf16,
        "b_bf16": b_bf16,
        "f8": f8,
        "scale": scale,
        "red_in": red_in,
    }
    outputs = {
        "dot": out_dot,
        "cvt": out_cvt.reshape(n, 4),
        "red": out_red,
    }
    return inputs, outputs


def _report(label, ref, out, *, atol, rtol):
    ref_f = ref.float()
    out_f = out.float()
    max_delta = (ref_f - out_f).abs().max().item()
    close = torch.isclose(ref_f, out_f, atol=atol, rtol=rtol)
    pct = close.float().mean().item() * 100.0
    passed = bool(close.all())
    print(f"  [{label}] max_delta={max_delta:.5f}, {pct:.2f}% close (atol={atol})")
    print(f"    ref  sample: {ref_f.reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out_f.reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta


def _check_dot2(inputs, outputs):
    ref = (inputs["a_bf16"].float() * inputs["b_bf16"].float()).sum(dim=1)
    return _report("dot2_f32_bf16", ref, outputs["dot"], atol=2e-2, rtol=2e-2)


def _check_convert(inputs, outputs):
    ref = inputs["f8"].float() * inputs["scale"][:, None]
    return _report(
        "cvt_scalef32_pk_bf16_fp8", ref, outputs["cvt"], atol=1e-1, rtol=2e-2
    )


def _check_reduce(inputs, outputs):
    ref = inputs["red_in"].float().sum().expand(WARP_SIZE)
    return _report("butterfly_reduce", ref, outputs["red"], atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("serialize_dot2", [True, False])
def test_dot2_f32_bf16(serialize_dot2):
    inputs, outputs = _run_primitives(serialize_dot2=serialize_dot2)
    passed, _ = _check_dot2(inputs, outputs)
    assert passed


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
def test_cvt_scalef32_pk_bf16_fp8():
    inputs, outputs = _run_primitives()
    passed, _ = _check_convert(inputs, outputs)
    assert passed


def test_butterfly_reduce():
    inputs, outputs = _run_primitives()
    passed, _ = _check_reduce(inputs, outputs)
    assert passed


# -------------------------------------------------------------------------
# Phase 2 -- gate_up FP8 fast path (BF16 activation, FP8 e4m3 weights)
# -------------------------------------------------------------------------
from aiter.ops.flydsl.warp_decode_moe import (  # noqa: E402
    flydsl_warp_decode_down_reduce,
    flydsl_warp_decode_gate_up,
    flydsl_warp_decode_gate_up_fp8act,
)

# name, B, HIDDEN, INTER, E, TOPK, w_scale_mode, scale_block (None | (BN, BK))
GATE_UP_CASES = [
    ("h1024_i64_e4_tk2_pertensor", 2, 1024, 64, 4, 2, "pertensor", None),
    ("h1024_i128_e8_tk2_pertoken", 1, 1024, 128, 8, 2, "pertoken", None),
    ("h512_i32_e2_tk1_kv8_pertensor", 1, 512, 32, 2, 1, "pertensor", None),
    ("h1024_i64_e4_tk2_block2d", 2, 1024, 64, 4, 2, "block2d", (128, 128)),
    ("h512_i32_e2_tk1_kv8_block2d", 1, 512, 32, 2, 1, "block2d", (64, 128)),
]


def _n_block2d_scales(rows, cols, bn, bk):
    return (rows // bn) * (cols // bk)


def _gen_gate_up(B, HIDDEN, INTER, E, TOPK, w_scale_mode, scale_block=None):
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260404)
    x = ((torch.rand((B, HIDDEN), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    w_gate = (
        (torch.rand((E, INTER, HIDDEN), generator=gen, device=device) - 0.5) * 0.5
    ).to(torch.float8_e4m3fn)
    w_up = (
        (torch.rand((E, INTER, HIDDEN), generator=gen, device=device) - 0.5) * 0.5
    ).to(torch.float8_e4m3fn)
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    if w_scale_mode == "pertensor":
        n_scale = 1
    elif w_scale_mode == "pertoken":
        n_scale = E * INTER
    else:  # block2d
        bn, bk = scale_block
        n_scale = _n_block2d_scales(E * INTER, HIDDEN, bn, bk)
    w_gate_scale = (
        torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5
    ).float()
    w_up_scale = (torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5).float()
    return x, w_gate, w_up, router_ids, w_gate_scale, w_up_scale


def _block2d_scale_matrix(p, e, rows_per_e, cols, bn, bk):
    """[rows_per_e, cols] scale matrix for expert ``e`` from flat Block2D scales."""
    device = p.device
    row_idx = e * rows_per_e + torch.arange(rows_per_e, device=device)
    col_blk = torch.arange(cols, device=device) // bk
    sidx = (row_idx // bn)[:, None] * (cols // bk) + col_blk[None, :]
    return p[sidx]


def _ref_gate_up(
    x,
    w_gate,
    w_up,
    router_ids,
    w_gate_scale,
    w_up_scale,
    w_scale_mode,
    scale_block=None,
):
    B, HIDDEN = x.shape
    E, INTER, _ = w_gate.shape
    TOPK = router_ids.shape[1]
    xf = x.float()
    wgf = w_gate.float()
    wuf = w_up.float()
    out = torch.empty(B, TOPK, INTER, dtype=torch.bfloat16, device=x.device)
    idx = torch.arange(INTER, device=x.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            if w_scale_mode == "block2d":
                bn, bk = scale_block
                sg = _block2d_scale_matrix(w_gate_scale, e, INTER, HIDDEN, bn, bk)
                su = _block2d_scale_matrix(w_up_scale, e, INTER, HIDDEN, bn, bk)
                gate = (wgf[e] * sg) @ xf[b]
                up = (wuf[e] * su) @ xf[b]
            else:
                gate = xf[b] @ wgf[e].T
                up = xf[b] @ wuf[e].T
                if w_scale_mode == "pertensor":
                    gate = gate * w_gate_scale[0]
                    up = up * w_up_scale[0]
                else:
                    rows = e * INTER + idx
                    gate = gate * w_gate_scale[rows]
                    up = up * w_up_scale[rows]
            silu = gate / (1.0 + torch.exp(-gate))
            out[b, k] = (silu * up).to(torch.bfloat16)
    return out


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _run_gate_up_case(case, *, cos_thresh=0.999):
    name, B, HIDDEN, INTER, E, TOPK, mode, scale_block = case
    print("=" * 78)
    print(f"[flydsl] warp-decode gate_up  case={name}")
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        B, HIDDEN, INTER, E, TOPK, mode, scale_block
    )
    out = flydsl_warp_decode_gate_up(
        x,
        w_gate,
        w_up,
        router_ids,
        wgs,
        wus,
        w_scale_mode=mode,
        scale_block=scale_block,
    )
    torch.cuda.synchronize()
    ref = _ref_gate_up(x, w_gate, w_up, router_ids, wgs, wus, mode, scale_block)
    cos = _cosine(ref, out)
    max_delta = (ref.float() - out.float()).abs().max().item()
    denom = ref.float().abs().max().item() + 1e-6
    passed = cos >= cos_thresh
    print(
        f"  cos_sim={cos:.6f} (thresh {cos_thresh}), "
        f"max_delta={max_delta:.4f} ({100*max_delta/denom:.2f}% of max)"
    )
    print(f"    ref  sample: {ref.float().reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out.float().reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, cos


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in GATE_UP_CASES])
def test_gate_up_fp8(case):
    passed, _ = _run_gate_up_case(case)
    assert passed


# -------------------------------------------------------------------------
# Phase 3 -- down_reduce FP8 fast path (BF16 intermediate, FP8 e4m3 weights)
# -------------------------------------------------------------------------
# name, B, INTER, HIDDEN, E, TOPK, w_scale_mode, scale_block (None | (BN, BK))
DOWN_CASES = [
    ("down_i1024_h64_e4_tk2_pertensor", 2, 1024, 64, 4, 2, "pertensor", None),
    ("down_i1024_h128_e8_tk2_pertoken", 1, 1024, 128, 8, 2, "pertoken", None),
    ("down_i512_h32_e2_tk1_kv8_pertensor", 1, 512, 32, 2, 1, "pertensor", None),
    ("down_i1024_h64_e4_tk2_block2d", 2, 1024, 64, 4, 2, "block2d", (128, 128)),
    ("down_i512_h32_e2_tk1_kv8_block2d", 1, 512, 32, 2, 1, "block2d", (64, 128)),
]


def _gen_down(B, INTER, HIDDEN, E, TOPK, w_scale_mode, scale_block=None):
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260405)
    inter = ((torch.rand((B, TOPK, INTER), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    w_down = (
        (torch.rand((E, HIDDEN, INTER), generator=gen, device=device) - 0.5) * 0.5
    ).to(torch.float8_e4m3fn)
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    router_wts = torch.rand((B, TOPK), generator=gen, device=device).float()
    router_wts = router_wts / router_wts.sum(dim=1, keepdim=True)
    if w_scale_mode == "pertensor":
        n_scale = 1
    elif w_scale_mode == "pertoken":
        n_scale = E * HIDDEN
    else:  # block2d
        bn, bk = scale_block
        n_scale = _n_block2d_scales(E * HIDDEN, INTER, bn, bk)
    w_down_scale = (
        torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5
    ).float()
    return inter, w_down, router_ids, router_wts, w_down_scale


def _ref_down(
    inter, w_down, router_ids, router_wts, w_down_scale, w_scale_mode, scale_block=None
):
    B, TOPK, INTER = inter.shape
    E, HIDDEN, _ = w_down.shape
    interf = inter.float()
    wdf = w_down.float()
    y = torch.zeros(B, HIDDEN, device=inter.device)
    idx = torch.arange(HIDDEN, device=inter.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            rw = router_wts[b, k]
            if w_scale_mode == "block2d":
                bn, bk = scale_block
                sd = _block2d_scale_matrix(w_down_scale, e, HIDDEN, INTER, bn, bk)
                dot = (wdf[e] * sd) @ interf[b, k]
                y[b] += rw * dot
            else:
                dot = interf[b, k] @ wdf[e].T
                if w_scale_mode == "pertensor":
                    ds = w_down_scale[0]
                else:
                    ds = w_down_scale[e * HIDDEN + idx]
                y[b] += dot * (rw * ds)
    return y.to(torch.bfloat16)


def _run_down_case(case, *, cos_thresh=0.999):
    name, B, INTER, HIDDEN, E, TOPK, mode, scale_block = case
    print("=" * 78)
    print(f"[flydsl] warp-decode down_reduce  case={name}")
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        B, INTER, HIDDEN, E, TOPK, mode, scale_block
    )
    out = flydsl_warp_decode_down_reduce(
        inter,
        w_down,
        router_ids,
        router_wts,
        wds,
        w_scale_mode=mode,
        scale_block=scale_block,
    )
    torch.cuda.synchronize()
    ref = _ref_down(inter, w_down, router_ids, router_wts, wds, mode, scale_block)
    cos = _cosine(ref, out)
    max_delta = (ref.float() - out.float()).abs().max().item()
    denom = ref.float().abs().max().item() + 1e-6
    passed = cos >= cos_thresh
    print(
        f"  cos_sim={cos:.6f} (thresh {cos_thresh}), "
        f"max_delta={max_delta:.4f} ({100*max_delta/denom:.2f}% of max)"
    )
    print(f"    ref  sample: {ref.float().reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out.float().reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, cos


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in DOWN_CASES])
def test_down_reduce_fp8(case):
    passed, _ = _run_down_case(case)
    assert passed


# -------------------------------------------------------------------------
# Phase E / G7 -- selectable s_nop-free ILP dot2 on FP8.  `dot2_acc>1` drains the
# per-lane dots through N independent f32 accumulators + one final add instead of
# the serialized `s_nop 2` chain (`dot2_acc=1`).  The two forms must agree to near
# bit-exactness (only f32 add *reassociation* differs) and both must match the fp32
# torch reference -- so these force `dot2_acc=4` on the same inputs as the baseline
# and cross-check.  Covers pertensor/pertoken/block2d (the block2d drain window is
# one K-block; the others span the whole K-range).
# -------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in GATE_UP_CASES])
def test_gate_up_fp8_ilp_dot2(case):
    name, B, HIDDEN, INTER, E, TOPK, mode, scale_block = case
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        B, HIDDEN, INTER, E, TOPK, mode, scale_block
    )
    out_base = flydsl_warp_decode_gate_up(
        x,
        w_gate,
        w_up,
        router_ids,
        wgs,
        wus,
        w_scale_mode=mode,
        scale_block=scale_block,
        dot2_acc=1,
    )
    out_ilp = flydsl_warp_decode_gate_up(
        x,
        w_gate,
        w_up,
        router_ids,
        wgs,
        wus,
        w_scale_mode=mode,
        scale_block=scale_block,
        dot2_acc=4,
    )
    torch.cuda.synchronize()
    ref = _ref_gate_up(x, w_gate, w_up, router_ids, wgs, wus, mode, scale_block)
    cos_ref = _cosine(ref, out_ilp)
    cos_base = _cosine(out_base, out_ilp)
    print(
        f"[fp8 gate_up ilp {name}] cos_vs_ref={cos_ref:.6f} cos_vs_base={cos_base:.6f}"
    )
    assert cos_ref >= 0.999, f"ilp gate_up {name}: cos_vs_ref={cos_ref:.6f}"
    assert cos_base >= 0.9999, f"ilp gate_up {name}: cos_vs_base={cos_base:.6f}"


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in DOWN_CASES])
def test_down_reduce_fp8_ilp_dot2(case):
    name, B, INTER, HIDDEN, E, TOPK, mode, scale_block = case
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        B, INTER, HIDDEN, E, TOPK, mode, scale_block
    )
    out_base = flydsl_warp_decode_down_reduce(
        inter,
        w_down,
        router_ids,
        router_wts,
        wds,
        w_scale_mode=mode,
        scale_block=scale_block,
        dot2_acc=1,
    )
    out_ilp = flydsl_warp_decode_down_reduce(
        inter,
        w_down,
        router_ids,
        router_wts,
        wds,
        w_scale_mode=mode,
        scale_block=scale_block,
        dot2_acc=4,
    )
    torch.cuda.synchronize()
    ref = _ref_down(inter, w_down, router_ids, router_wts, wds, mode, scale_block)
    cos_ref = _cosine(ref, out_ilp)
    cos_base = _cosine(out_base, out_ilp)
    print(f"[fp8 down ilp {name}] cos_vs_ref={cos_ref:.6f} cos_vs_base={cos_base:.6f}")
    assert cos_ref >= 0.999, f"ilp down {name}: cos_vs_ref={cos_ref:.6f}"
    assert cos_base >= 0.9999, f"ilp down {name}: cos_vs_base={cos_base:.6f}"


# -------------------------------------------------------------------------
# Phase D / G5 -- Split-K down (see SILOTIGER-667-plan-Split-K.md).  Each split-K
# wave covers a disjoint INTER sub-range and atomic-adds its FP32 partial into a
# caller-zeroed accumulator; the result must match the non-split (split_k=1) path
# and the fp32 reference (only atomic reassociation differs).  num_iter =
# INTER/(64*kvector) must be divisible by split_k (kv=16 when INTER%1024==0).
# name, B, INTER, HIDDEN, E, TOPK, split_k.
DOWN_SPLITK_CASES = [
    ("splitk_i2048_h128_e8_tk2_sk2", 1, 2048, 128, 8, 2, 2),
    ("splitk_i4096_h64_e8_tk2_sk4", 1, 4096, 64, 8, 2, 4),
]


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in DOWN_SPLITK_CASES])
def test_down_reduce_split_k(case):
    name, B, INTER, HIDDEN, E, TOPK, split_k = case
    inter, w_down, rid, rwt, wds = _gen_down(B, INTER, HIDDEN, E, TOPK, "pertensor")
    base = flydsl_warp_decode_down_reduce(
        inter, w_down, rid, rwt, wds, w_scale_mode="pertensor"
    )
    got = flydsl_warp_decode_down_reduce(
        inter, w_down, rid, rwt, wds, w_scale_mode="pertensor", split_k=split_k
    )
    torch.cuda.synchronize()
    ref = _ref_down(inter, w_down, rid, rwt, wds, "pertensor")
    cos_ref = _cosine(ref, got)
    cos_base = _cosine(base, got)
    print(f"[splitk {name}] cos_vs_ref={cos_ref:.6f} cos_vs_base={cos_base:.6f}")
    assert cos_ref >= 0.999, f"split_k {name}: cos_vs_ref={cos_ref:.6f}"
    assert cos_base >= 0.999, f"split_k {name}: cos_vs_base={cos_base:.6f}"


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
def test_down_reduce_split_k_auto():
    """split_k='auto' must match the non-split path regardless of the CU gate's pick.

    Small grid (B=1, HIDDEN=128 -> base_grid=64 with kh=2) should trigger a >1
    factor; we don't assert the exact k (it's CU-count dependent) but do assert
    the auto path is numerically faithful to the plain bf16 store."""
    from aiter.ops.flydsl.warp_decode_moe import _auto_split_k_down, _cu_count

    B, INTER, HIDDEN, E, TOPK = 1, 2048, 128, 8, 2
    inter, w_down, rid, rwt, wds = _gen_down(B, INTER, HIDDEN, E, TOPK, "pertensor")
    base = flydsl_warp_decode_down_reduce(
        inter, w_down, rid, rwt, wds, w_scale_mode="pertensor"
    )
    got = flydsl_warp_decode_down_reduce(
        inter, w_down, rid, rwt, wds, w_scale_mode="pertensor", split_k="auto"
    )
    torch.cuda.synchronize()
    cos_base = _cosine(base, got)
    picked = _auto_split_k_down(B, HIDDEN, 2, INTER, 16, 0)
    print(
        f"[splitk auto] picked_k={picked} cu={_cu_count(0)} cos_vs_base={cos_base:.6f}"
    )
    assert cos_base >= 0.999, f"split_k auto: cos_vs_base={cos_base:.6f}"


def test_auto_split_k_gate_logic():
    """Unit-test the CU-count gate: divisibility + base_grid*k <= CuCount, else 1."""
    from aiter.ops.flydsl.warp_decode_moe import _auto_split_k_down

    # base_grid = B*(HIDDEN/kh) = 1*(128/2) = 64; num_iter = 2048/(64*16) = 2.
    # -> only k=2 divides num_iter; picked iff 64*2 <= CuCount (true on gfx950).
    assert _auto_split_k_down(1, 128, 2, 2048, 16, 0) in (1, 2)
    # Saturated grid: base_grid huge (DeepSeek-like) must stay at 1 (off).
    assert _auto_split_k_down(1, 7168, 2, 2048, 16, 0) == 1
    # num_iter=2 not divisible by 4 or 8, so only 2 is ever a candidate here.
    assert _auto_split_k_down(1, 8, 2, 2048, 16, 0) in (1, 2)


# -------------------------------------------------------------------------
# Phase B -- down_reduce MXFP4 (BF16 intermediate, FP4 e2m1 + E8M0 block scale)
# -------------------------------------------------------------------------
from aiter.ops.flydsl.warp_decode_moe import (  # noqa: E402
    flydsl_warp_decode_down_reduce_fp4,
)

# The MXFP4 codebook (E2M1); index = 4-bit nibble. Mirrors the LUT hardcoded in
# aiter.utility.fp4_utils.mxfp4_to_f32 (which we can't import here because this
# torch build lacks torch.float4_e2m1fn_x2, referenced by that helper).
_MXFP4_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# MXFP4 uses a fixed Block2D<1,32> E8M0 scale (BK=32 by spec). INTER must be a
# multiple of 64*8=512 (one i32 = 8 FP4 = one weight dword/lane/iter).
# name, B, INTER, HIDDEN, E, TOPK
# name, B, INTER, HIDDEN, E, TOPK, kvector (explicit to cover 8/16/32 builder paths;
# the shipped default is always 8 -- see pick_kvector_fp4).
DOWN_FP4_CASES = [
    ("down_fp4_i512_h256_e8_tk2_kv8", 1, 512, 256, 8, 2, 8),
    ("down_fp4_i1024_h128_e8_tk4_kv16", 2, 1024, 128, 8, 4, 16),
    ("down_fp4_i2048_h128_e8_tk4_kv32", 1, 2048, 128, 8, 4, 32),
    # E=256 keeps the real DeepSeek expert count in the fast suite so the K3
    # Tier-1 w_row*(INTER//8) offset path stays exercised (small dims -> light pool).
    ("down_fp4_i512_h128_e256_tk8_kv8", 1, 512, 128, 256, 8, 8),
]
_MXFP4_BK = 32


def _gen_down_fp4(B, INTER, HIDDEN, E, TOPK):
    """MXFP4 down inputs + fp32 dequantized reference weights.

    FP4 codebook weights with power-of-two E8M0 block scales (both exactly
    representable in bf16 after the scaled convert), so the only error source is
    bf16 rounding of the intermediate + dot2 summation order.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260810)
    rows = E * HIDDEN
    scale_cols = INTER // _MXFP4_BK
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=device)

    codes = torch.randint(
        0, 16, (rows, INTER), generator=gen, device=device, dtype=torch.int32
    )
    # E8M0 bytes in a moderate power-of-two range (2^-4 .. 2^4) => exact.
    sbytes = torch.randint(
        123, 132, (rows, scale_cols), generator=gen, device=device, dtype=torch.int32
    )
    blk = torch.pow(2.0, sbytes.float() - 127.0)
    w_deq = (lut[codes.long()] * blk.repeat_interleave(_MXFP4_BK, dim=1)).reshape(
        E, HIDDEN, INTER
    )

    # Pack 2 FP4/byte (low nibble = even element) -> uint8 [E, HIDDEN, INTER//2].
    packed = ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8)
    w_down = packed.reshape(E, HIDDEN, INTER // 2).contiguous()
    w_scale = sbytes.to(torch.uint8).reshape(rows, scale_cols).contiguous()

    inter = ((torch.rand((B, TOPK, INTER), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    router_ids[0, 0] = E - 1  # exercise the max weight-row offset
    router_wts = torch.rand((B, TOPK), generator=gen, device=device).float()
    router_wts = router_wts / router_wts.sum(dim=1, keepdim=True)
    return inter, w_down, w_scale, router_ids, router_wts, w_deq


def _ref_down_fp4(inter, w_deq, router_ids, router_wts):
    B, TOPK, _ = inter.shape
    _, HIDDEN, _ = w_deq.shape
    interf = inter.float()
    y = torch.zeros(B, HIDDEN, device=inter.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            y[b] += float(router_wts[b, k]) * (interf[b, k] @ w_deq[e].T)
    return y.to(torch.bfloat16)


def _run_down_fp4_case(case, *, cos_thresh=0.99):
    name, B, INTER, HIDDEN, E, TOPK, kvector = case
    print("=" * 78)
    print(f"[flydsl] warp-decode down_reduce MXFP4  case={name}")
    inter, w_down, w_scale, router_ids, router_wts, w_deq = _gen_down_fp4(
        B, INTER, HIDDEN, E, TOPK
    )
    out = flydsl_warp_decode_down_reduce_fp4(
        inter,
        w_down,
        router_ids,
        router_wts,
        w_scale,
        scale_block=(1, _MXFP4_BK),
        kvector=kvector,
    )
    torch.cuda.synchronize()
    ref = _ref_down_fp4(inter, w_deq, router_ids, router_wts)
    cos = _cosine(ref, out)
    max_delta = (ref.float() - out.float()).abs().max().item()
    denom = ref.float().abs().max().item() + 1e-6
    passed = cos >= cos_thresh
    print(
        f"  cos_sim={cos:.6f} (thresh {cos_thresh}), "
        f"max_delta={max_delta:.4f} ({100*max_delta/denom:.2f}% of max)"
    )
    print(f"    ref  sample: {ref.float().reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out.float().reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, cos


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in DOWN_FP4_CASES])
def test_down_reduce_fp4(case):
    passed, _ = _run_down_fp4_case(case)
    assert passed


# -------------------------------------------------------------------------
# Phase B -- gate_up MXFP4 (BF16 activation, FP4 e2m1 + E8M0 block scale)
# -------------------------------------------------------------------------
from aiter.ops.flydsl.warp_decode_moe import (  # noqa: E402
    flydsl_warp_decode_gate_up_fp4,
)

# name, B, HIDDEN, INTER, E, TOPK. HIDDEN (the contraction) must be a multiple
# of 64*8=512 (FP4 fast path).
# name, B, HIDDEN, INTER, E, TOPK, kvector (HIDDEN is the contraction; explicit
# kvector covers the 8/16/32 builder paths, shipped default is always 8).
GATE_UP_FP4_CASES = [
    ("gate_up_fp4_h512_i256_e8_tk2_kv8", 1, 512, 256, 8, 2, 8),
    ("gate_up_fp4_h1024_i128_e8_tk4_kv16", 2, 1024, 128, 8, 4, 16),
    ("gate_up_fp4_h2048_i128_e8_tk4_kv32", 1, 2048, 128, 8, 4, 32),
    # E=256 (real expert count) keeps the K3 Tier-1 w_row*(HIDDEN//8) path covered.
    ("gate_up_fp4_h512_i128_e256_tk8_kv8", 1, 512, 128, 256, 8, 8),
]


def _gen_gate_up_fp4(B, HIDDEN, INTER, E, TOPK):
    """MXFP4 gate_up inputs + fp32 dequantized reference weights.

    Mirrors ``_gen_down_fp4`` but with two weight streams (gate, up), each with
    its own E8M0 block scale; contraction dim is HIDDEN.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    rows = E * INTER
    scale_cols = HIDDEN // _MXFP4_BK
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=device)

    def _one():
        codes = torch.randint(
            0, 16, (rows, HIDDEN), generator=gen, device=device, dtype=torch.int32
        )
        sbytes = torch.randint(
            123,
            132,
            (rows, scale_cols),
            generator=gen,
            device=device,
            dtype=torch.int32,
        )
        blk = torch.pow(2.0, sbytes.float() - 127.0)
        deq = (lut[codes.long()] * blk.repeat_interleave(_MXFP4_BK, dim=1)).reshape(
            E, INTER, HIDDEN
        )
        packed = ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8)
        w = packed.reshape(E, INTER, HIDDEN // 2).contiguous()
        s = sbytes.to(torch.uint8).reshape(rows, scale_cols).contiguous()
        return w, s, deq

    w_gate, gs, w_gate_deq = _one()
    w_up, us, w_up_deq = _one()

    x = ((torch.rand((B, HIDDEN), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    router_ids[0, 0] = E - 1  # exercise the max weight-row offset
    return x, w_gate, w_up, gs, us, router_ids, w_gate_deq, w_up_deq


def _ref_gate_up_fp4(x, w_gate_deq, w_up_deq, router_ids):
    B, HIDDEN = x.shape
    E, INTER, _ = w_gate_deq.shape
    TOPK = router_ids.shape[1]
    xf = x.float()
    out = torch.empty(B, TOPK, INTER, dtype=torch.bfloat16, device=x.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            gate = xf[b] @ w_gate_deq[e].T
            up = xf[b] @ w_up_deq[e].T
            silu = gate / (1.0 + torch.exp(-gate))
            out[b, k] = (silu * up).to(torch.bfloat16)
    return out


def _run_gate_up_fp4_case(case, *, cos_thresh=0.99):
    name, B, HIDDEN, INTER, E, TOPK, kvector = case
    print("=" * 78)
    print(f"[flydsl] warp-decode gate_up MXFP4  case={name}")
    x, w_gate, w_up, gs, us, router_ids, wg_deq, wu_deq = _gen_gate_up_fp4(
        B, HIDDEN, INTER, E, TOPK
    )
    out = flydsl_warp_decode_gate_up_fp4(
        x,
        w_gate,
        w_up,
        router_ids,
        gs,
        us,
        scale_block=(1, _MXFP4_BK),
        kvector=kvector,
    )
    torch.cuda.synchronize()
    ref = _ref_gate_up_fp4(x, wg_deq, wu_deq, router_ids)
    cos = _cosine(ref, out)
    max_delta = (ref.float() - out.float()).abs().max().item()
    denom = ref.float().abs().max().item() + 1e-6
    passed = cos >= cos_thresh
    print(
        f"  cos_sim={cos:.6f} (thresh {cos_thresh}), "
        f"max_delta={max_delta:.4f} ({100*max_delta/denom:.2f}% of max)"
    )
    print(f"    ref  sample: {ref.float().reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out.float().reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, cos


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in GATE_UP_FP4_CASES])
def test_gate_up_fp4(case):
    passed, _ = _run_gate_up_fp4_case(case)
    assert passed


# -------------------------------------------------------------------------
# Phase C -- BF16 weight path (unquantized correctness oracle)
# -------------------------------------------------------------------------
from aiter.ops.flydsl.warp_decode_moe import (  # noqa: E402
    flydsl_warp_decode_down_reduce_bf16,
    flydsl_warp_decode_gate_up_bf16,
)

# name, B, {HIDDEN|INTER contraction first}, ..., E, TOPK. Contraction dim must be
# a multiple of 64*8=512. These sweep small + real (E=256) expert counts; the BF16
# path has no quantization, so cos should be ~1.0 (only bf16 dot2 rounding).
BF16_GATE_UP_CASES = [
    ("bf16_gate_up_h512_i256_e8_tk2", 1, 512, 256, 8, 2),
    ("bf16_gate_up_h1024_i128_e32_tk4", 2, 1024, 128, 32, 4),
    ("bf16_gate_up_h512_i128_e256_tk8", 1, 512, 128, 256, 8),
]
BF16_DOWN_CASES = [
    ("bf16_down_i512_h256_e8_tk2", 1, 512, 256, 8, 2),
    ("bf16_down_i1024_h128_e32_tk4", 2, 1024, 128, 32, 4),
    ("bf16_down_i512_h128_e256_tk8", 1, 512, 128, 256, 8),
]


def _gen_bf16_gate_up(B, HIDDEN, INTER, E, TOPK):
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    scale = 0.1  # keep magnitudes small so bf16 rounding stays tight
    x = (torch.randn((B, HIDDEN), generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    w_gate = (torch.randn((E, INTER, HIDDEN), generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    w_up = (torch.randn((E, INTER, HIDDEN), generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    router_ids[0, 0] = E - 1  # exercise the max weight-row offset
    return x, w_gate, w_up, router_ids


def _ref_bf16_gate_up(x, w_gate, w_up, router_ids):
    B, HIDDEN = x.shape
    _, INTER, _ = w_gate.shape
    TOPK = router_ids.shape[1]
    xf = x.float()
    out = torch.zeros(B, TOPK, INTER, device=x.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            gate = xf[b] @ w_gate[e].float().T
            up = xf[b] @ w_up[e].float().T
            silu = gate * torch.sigmoid(gate)
            out[b, k] = silu * up
    return out.to(torch.bfloat16)


def _gen_bf16_down(B, INTER, HIDDEN, E, TOPK):
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    scale = 0.1
    inter = (torch.randn((B, TOPK, INTER), generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    w_down = (torch.randn((E, HIDDEN, INTER), generator=gen, device=device) * scale).to(
        torch.bfloat16
    )
    router_ids = torch.randint(
        0, E, (B, TOPK), generator=gen, device=device, dtype=torch.int32
    )
    router_ids[0, 0] = E - 1
    router_wts = torch.rand((B, TOPK), generator=gen, device=device).float()
    router_wts = router_wts / router_wts.sum(dim=1, keepdim=True)
    return inter, w_down, router_ids, router_wts


def _ref_bf16_down(inter, w_down, router_ids, router_wts):
    B, TOPK, _ = inter.shape
    _, HIDDEN, _ = w_down.shape
    interf = inter.float()
    y = torch.zeros(B, HIDDEN, device=inter.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(router_ids[b, k])
            y[b] += float(router_wts[b, k]) * (interf[b, k] @ w_down[e].float().T)
    return y.to(torch.bfloat16)


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in BF16_GATE_UP_CASES])
def test_gate_up_bf16(case):
    name, B, HIDDEN, INTER, E, TOPK = case
    x, w_gate, w_up, router_ids = _gen_bf16_gate_up(B, HIDDEN, INTER, E, TOPK)
    out = flydsl_warp_decode_gate_up_bf16(x, w_gate, w_up, router_ids)
    torch.cuda.synchronize()
    ref = _ref_bf16_gate_up(x, w_gate, w_up, router_ids)
    cos = _cosine(ref, out)
    print(f"[bf16 gate_up {name}] cos={cos:.6f}")
    assert cos >= 0.99, f"bf16 gate_up {name}: cos={cos:.6f}"


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in BF16_DOWN_CASES])
def test_down_reduce_bf16(case):
    name, B, INTER, HIDDEN, E, TOPK = case
    inter, w_down, router_ids, router_wts = _gen_bf16_down(B, INTER, HIDDEN, E, TOPK)
    out = flydsl_warp_decode_down_reduce_bf16(inter, w_down, router_ids, router_wts)
    torch.cuda.synchronize()
    ref = _ref_bf16_down(inter, w_down, router_ids, router_wts)
    cos = _cosine(ref, out)
    print(f"[bf16 down {name}] cos={cos:.6f}")
    assert cos >= 0.99, f"bf16 down {name}: cos={cos:.6f}"


# -------------------------------------------------------------------------
# gfx942 scalar-f32 fallback (SILOTIGER-667 Phase C / G4).  The scalar path
# (`use_dot2=False`) replaces `v_dot2_f32_bf16` (a gfx950 instruction) with pure
# f32 unpack+FMA, so it compiles and runs on every AMD arch.  We can't test real
# gfx942 hardware here, but forcing `use_dot2=False` on gfx950 exercises the exact
# fallback math: we assert (a) it still matches the fp32 torch reference and (b) it
# matches the dot2 path to near bit-exactness (only f32 add reassociation differs).
# The auto path (`use_dot2=None`) picks dot2 on gfx950, so these forced cases are
# the only way to cover the fallback without MI300 hardware.
# -------------------------------------------------------------------------
@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in BF16_GATE_UP_CASES])
def test_gate_up_bf16_scalar_fallback(case):
    name, B, HIDDEN, INTER, E, TOPK = case
    x, w_gate, w_up, router_ids = _gen_bf16_gate_up(B, HIDDEN, INTER, E, TOPK)
    out_scalar = flydsl_warp_decode_gate_up_bf16(
        x, w_gate, w_up, router_ids, use_dot2=False
    )
    out_dot2 = flydsl_warp_decode_gate_up_bf16(
        x, w_gate, w_up, router_ids, use_dot2=True
    )
    torch.cuda.synchronize()
    ref = _ref_bf16_gate_up(x, w_gate, w_up, router_ids)
    cos_ref = _cosine(ref, out_scalar)
    cos_dot2 = _cosine(out_dot2, out_scalar)
    print(
        f"[bf16 gate_up scalar {name}] cos_vs_ref={cos_ref:.6f} "
        f"cos_vs_dot2={cos_dot2:.6f}"
    )
    assert cos_ref >= 0.99, f"scalar gate_up {name}: cos_vs_ref={cos_ref:.6f}"
    assert cos_dot2 >= 0.9999, f"scalar gate_up {name}: cos_vs_dot2={cos_dot2:.6f}"


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in BF16_DOWN_CASES])
def test_down_reduce_bf16_scalar_fallback(case):
    name, B, INTER, HIDDEN, E, TOPK = case
    inter, w_down, router_ids, router_wts = _gen_bf16_down(B, INTER, HIDDEN, E, TOPK)
    out_scalar = flydsl_warp_decode_down_reduce_bf16(
        inter, w_down, router_ids, router_wts, use_dot2=False
    )
    out_dot2 = flydsl_warp_decode_down_reduce_bf16(
        inter, w_down, router_ids, router_wts, use_dot2=True
    )
    torch.cuda.synchronize()
    ref = _ref_bf16_down(inter, w_down, router_ids, router_wts)
    cos_ref = _cosine(ref, out_scalar)
    cos_dot2 = _cosine(out_dot2, out_scalar)
    print(
        f"[bf16 down scalar {name}] cos_vs_ref={cos_ref:.6f} "
        f"cos_vs_dot2={cos_dot2:.6f}"
    )
    assert cos_ref >= 0.99, f"scalar down {name}: cos_vs_ref={cos_ref:.6f}"
    assert cos_dot2 >= 0.9999, f"scalar down {name}: cos_vs_dot2={cos_dot2:.6f}"


# -------------------------------------------------------------------------
# BF16-oracle cross-check: run the FP4 kernel and the BF16 kernel on the *same*
# logical weights (FP4's e8m0 scales are powers of two + LUT values exact => the
# fp32 dequant is bf16-exact, so the BF16 kernel fed `w_deq.to(bf16)` computes
# exactly what the FP4 kernel converts internally).  The two outputs must agree to
# near bit-exactness (only the f32 dot2 *reassociation* differs -- FP4 uses the G7
# 4-accumulator drain, BF16 a serial chain), which isolates any FP4 convert/scale
# bug from the reduce/routing.  Includes the real E=256 shape.
# -------------------------------------------------------------------------
XCHECK_DOWN_CASES = [
    ("xcheck_down_i512_h256_e8_tk2", 1, 512, 256, 8, 2),
    ("xcheck_down_i2048_h128_e256_tk8", 1, 2048, 128, 256, 8),
]
XCHECK_GATE_UP_CASES = [
    ("xcheck_gate_up_h512_i256_e8_tk2", 1, 512, 256, 8, 2),
    ("xcheck_gate_up_h512_i128_e256_tk8", 1, 512, 128, 256, 8),
]


@pytest.mark.parametrize("case", [pytest.param(c, id=c[0]) for c in XCHECK_DOWN_CASES])
def test_down_fp4_matches_bf16_oracle(case):
    name, B, INTER, HIDDEN, E, TOPK = case
    inter, w_down, w_scale, rid, rwt, w_deq = _gen_down_fp4(B, INTER, HIDDEN, E, TOPK)
    out_fp4 = flydsl_warp_decode_down_reduce_fp4(
        inter, w_down, rid, rwt, w_scale, scale_block=(1, _MXFP4_BK)
    )
    out_bf16 = flydsl_warp_decode_down_reduce_bf16(
        inter, w_deq.to(torch.bfloat16).contiguous(), rid, rwt
    )
    torch.cuda.synchronize()
    cos_xc = _cosine(out_bf16, out_fp4)
    ref = _ref_down_fp4(inter, w_deq, rid, rwt)
    cos_ref = _cosine(ref, out_fp4)
    print(f"[xcheck down {name}] fp4~bf16 cos={cos_xc:.6f}  fp4~fp32 cos={cos_ref:.6f}")
    assert (
        cos_xc >= 0.999
    ), f"down {name}: FP4 diverges from BF16 oracle (cos={cos_xc:.6f})"


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c[0]) for c in XCHECK_GATE_UP_CASES]
)
def test_gate_up_fp4_matches_bf16_oracle(case):
    name, B, HIDDEN, INTER, E, TOPK = case
    x, w_gate, w_up, gs, us, rid, wg_deq, wu_deq = _gen_gate_up_fp4(
        B, HIDDEN, INTER, E, TOPK
    )
    out_fp4 = flydsl_warp_decode_gate_up_fp4(
        x, w_gate, w_up, rid, gs, us, scale_block=(1, _MXFP4_BK)
    )
    out_bf16 = flydsl_warp_decode_gate_up_bf16(
        x,
        wg_deq.to(torch.bfloat16).contiguous(),
        wu_deq.to(torch.bfloat16).contiguous(),
        rid,
    )
    torch.cuda.synchronize()
    cos_xc = _cosine(out_bf16, out_fp4)
    ref = _ref_gate_up_fp4(x, wg_deq, wu_deq, rid)
    cos_ref = _cosine(ref, out_fp4)
    print(
        f"[xcheck gate_up {name}] fp4~bf16 cos={cos_xc:.6f}  fp4~fp32 cos={cos_ref:.6f}"
    )
    assert (
        cos_xc >= 0.999
    ), f"gate_up {name}: FP4 diverges from BF16 oracle (cos={cos_xc:.6f})"


# -------------------------------------------------------------------------
# Real expert-count regression (SILOTIGER-667-plan-10082026 Phase A / G1)
# -------------------------------------------------------------------------
# GATE_UP_CASES / DOWN_CASES all use E<=8, which never exercises a large
# weight-row offset. These cases run the real ticket expert counts and force
# the *max-id* expert (E-1, the largest weight-row offset) into the routing,
# locking in offset-arithmetic correctness at production expert counts:
#   * DeepSeek-V3   (E=256, H7168/I2048/TOPK8)  -- the largest ticket tensor.
#   * Qwen3Next TP1 (E=512, H2048/I512/TOPK10)  -- the largest ticket E.
# Verified cos=1.0 on gfx950; the only FlyDSL addressing limit (buffer_load's
# i32 dword index) is reached solely by >8 GB weight tensors, far above any
# ticket shape (see the plan's Phase A / ?9). References are computed for the
# routed (b,k) rows only, so HBM use is dominated by the FP8 weight tensors
# (~3.5 GB for DeepSeek; guarded below).
# name -> dict(B, HIDDEN, INTER, E, TOPK)  (HIDDEN = model/output dim, INTER =
# expert intermediate / contraction; matches the _gen_* helpers' semantics).
REAL_E_CASES = [
    ("deepseek_v3_e256", dict(B=1, HIDDEN=7168, INTER=2048, E=256, TOPK=8)),
    ("qwen3next_tp1_e512", dict(B=1, HIDDEN=2048, INTER=512, E=512, TOPK=10)),
]
_REAL_E_MIN_FREE_GB = 16.0


def _skip_if_low_hbm():
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1e9
    if free_gb < _REAL_E_MIN_FREE_GB:
        pytest.skip(
            f"needs >= {_REAL_E_MIN_FREE_GB:.0f} GB free HBM for the real-E weight "
            f"tensors (have {free_gb:.1f} GB)"
        )


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(d, id=n) for n, d in REAL_E_CASES])
def test_gate_up_fp8_real_expert_count(case):
    """gate_up at real E with the max-offset expert forced (G1 regression)."""
    _skip_if_low_hbm()
    d = case
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        d["B"], d["HIDDEN"], d["INTER"], d["E"], d["TOPK"], "pertensor"
    )
    router_ids[0, 0] = d["E"] - 1  # largest weight-row offset
    out = flydsl_warp_decode_gate_up(
        x, w_gate, w_up, router_ids, wgs, wus, w_scale_mode="pertensor"
    ).float()
    torch.cuda.synchronize()
    # Compact reference: only the routed (b,k) rows (avoid materialising the
    # whole [E, INTER, HIDDEN] weight tensor in fp32).
    xf = x.float()
    worst_cos = 1.0
    for b in range(d["B"]):
        for k in range(d["TOPK"]):
            e = int(router_ids[b, k])
            g = (xf[b] @ w_gate[e].float().T) * float(wgs[0])
            u = (xf[b] @ w_up[e].float().T) * float(wus[0])
            ref = (g / (1.0 + torch.exp(-g))) * u
            cos = _cosine(ref, out[b, k])
            worst_cos = min(worst_cos, cos)
    print(f"[real-E gate_up] E={d['E']} worst per-(b,k) cos={worst_cos:.6f}")
    assert worst_cos >= 0.999


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
@pytest.mark.parametrize("case", [pytest.param(d, id=n) for n, d in REAL_E_CASES])
def test_down_reduce_fp8_real_expert_count(case):
    """down_reduce at real E with the max-offset expert forced (G1 regression)."""
    _skip_if_low_hbm()
    d = case
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        d["B"], d["INTER"], d["HIDDEN"], d["E"], d["TOPK"], "pertensor"
    )
    router_ids[0, 0] = d["E"] - 1  # largest weight-row offset
    out = flydsl_warp_decode_down_reduce(
        inter, w_down, router_ids, router_wts, wds, w_scale_mode="pertensor"
    ).float()
    torch.cuda.synchronize()
    ref = torch.zeros(d["B"], d["HIDDEN"], device=inter.device)
    for b in range(d["B"]):
        for k in range(d["TOPK"]):
            e = int(router_ids[b, k])
            dot = inter[b, k].float() @ w_down[e].float().T
            ref[b] += float(router_wts[b, k]) * float(wds[0]) * dot
    cos = _cosine(ref, out)
    print(f"[real-E down] E={d['E']} cos={cos:.6f}")
    assert cos >= 0.999


# -------------------------------------------------------------------------
# Perf sweep -- combined correctness + benchmark (SILOTIGER-667 ?2 standard)
# -------------------------------------------------------------------------
# Realistic decode shapes (weights >> last-level cache). B, HIDDEN, INTER, E,
# TOPK, mode, scale_block.  DeepSeek-V3-ish (H7168/I2048/TOPK8) is the headline.
GATE_UP_PERF_SHAPES = [
    (1, 7168, 2048, 8, 8, "pertensor", None),
    (4, 7168, 2048, 8, 8, "pertensor", None),
    (1, 7168, 2048, 8, 8, "pertoken", None),
    (1, 7168, 2048, 8, 8, "block2d", (128, 128)),
    (1, 4096, 1024, 8, 8, "pertensor", None),
]
# B, INTER, HIDDEN, E, TOPK, mode, scale_block.
DOWN_PERF_SHAPES = [
    (1, 2048, 7168, 8, 8, "pertensor", None),
    (4, 2048, 7168, 8, 8, "pertensor", None),
    (1, 2048, 7168, 8, 8, "pertoken", None),
    (1, 2048, 7168, 8, 8, "block2d", (128, 128)),
    (1, 1024, 4096, 8, 8, "pertensor", None),
]
# MXFP4 down A/B sweep: B in {1,2,4,8} tests the FP4-vs-FP8 crossover (expect
# ~1.2-1.5x at B>=2, neutral at B=1). INTER must be a multiple of 512 (FP4 fast
# path). B, INTER, HIDDEN, E, TOPK.  DeepSeek-V3 down is the headline.
DOWN_FP4_PERF_SHAPES = [
    (1, 2048, 7168, 8, 8),
    (2, 2048, 7168, 8, 8),
    (4, 2048, 7168, 8, 8),
    (8, 2048, 7168, 8, 8),
    (1, 1536, 3072, 8, 8),  # MiniMax
    (1, 512, 2048, 512, 10),  # Qwen3Next-TP1 (E=512, TOPK=10)
]


def _timing_kwargs(timing: str) -> dict:
    """Map a --timing choice to run_perftest kwargs (see SILOTIGER-667 ?2).

    device     : torch-profiler *device* time only (pure kernel), IQR-trimmed
                 when num_iters > 30. The reported headline BW.
    cuda_event : wall-clock mean per launch -- includes host dispatch (the
                 entry-point's per-call ptr_arg + current_stream cost, real at
                 B=1, ~20 us kernels).
    graph      : CUDA-graph replay + device time (lowest host overhead).
    """
    if timing == "cuda_event":
        return {"use_cuda_event": True}
    if timing == "graph":
        return {"testGraph": True}
    return {}  # device (default)


def _rotate_for(*tensors) -> int:
    """Pick ``num_rotate_args`` for a cold-HBM-read measurement without OOM.

    SILOTIGER-667 ?2 wants each timed iter to stream weights from HBM (not reuse
    cache). ``run_perftest``'s default auto-rotation deep-copies enough input
    sets to fill cache -- fine for tiny tensors, but it OOMs on 100 MB+ FP8
    weight tensors. When the weight set already dwarfs the last-level cache (any
    realistic decode shape) reads are cold with no rotation at all, so we return
    a tiny fixed count; only genuinely small working sets fall back to auto (0).
    """
    nbytes = sum(
        t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor)
    )
    llc = 256 * 1024 * 1024  # ~MI350 MALL; above this, reads are cold w/o copies
    if nbytes >= llc:
        return 2  # cold already; 1 deep-copy keeps the profiler's rotation happy
    return 0  # small: let run_perftest auto-fill cache to force cold reads


@benchmark()
def bench_gate_up(
    B, HIDDEN, INTER, E, TOPK, mode, timing, num_iters, num_warmup, scale_block=None
):
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        B, HIDDEN, INTER, E, TOPK, mode, scale_block
    )
    # Faithful to the real call: pre-allocate and pass the output buffer.
    out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=x.device)
    ref = _ref_gate_up(
        x, w_gate, w_up, router_ids, wgs, wus, mode, scale_block
    )  # not timed

    fn = lambda: flydsl_warp_decode_gate_up(  # noqa: E731
        x,
        w_gate,
        w_up,
        router_ids,
        wgs,
        wus,
        w_scale_mode=mode,
        scale_block=scale_block,
        out=out,
    )
    got, us = run_perftest(
        fn,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=_rotate_for(x, w_gate, w_up),
        **_timing_kwargs(timing),
    )
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.05,
        msg=f"gate_up {mode}",
        printLog=False,
    )
    assert _cosine(ref, got) >= 0.999, f"gate_up {mode}: correctness regression"
    # B3: single source of truth for the byte/FLOP model (weight_stream, FP8 weights).
    m = compute_metrics("gate_up", B, HIDDEN, INTER, TOPK, "fp8", us)
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": m["TFLOPS"],
        "TB/s": m["TB/s"],
        "%peak": m["%peak"],
        "err": err,
    }


@benchmark()
def bench_down(
    B, INTER, HIDDEN, E, TOPK, mode, timing, num_iters, num_warmup, scale_block=None
):
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        B, INTER, HIDDEN, E, TOPK, mode, scale_block
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=inter.device)
    ref = _ref_down(
        inter, w_down, router_ids, router_wts, wds, mode, scale_block
    )  # not timed

    fn = lambda: flydsl_warp_decode_down_reduce(  # noqa: E731
        inter,
        w_down,
        router_ids,
        router_wts,
        wds,
        w_scale_mode=mode,
        scale_block=scale_block,
        out=out,
    )
    got, us = run_perftest(
        fn,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=_rotate_for(inter, w_down),
        **_timing_kwargs(timing),
    )
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.05,
        msg=f"down {mode}",
        printLog=False,
    )
    assert _cosine(ref, got) >= 0.999, f"down {mode}: correctness regression"
    # B3: single source of truth for the byte/FLOP model (weight_stream, FP8 weights).
    m = compute_metrics("down", B, HIDDEN, INTER, TOPK, "fp8", us)
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": m["TFLOPS"],
        "TB/s": m["TB/s"],
        "%peak": m["%peak"],
        "err": err,
    }


@benchmark()
def bench_down_fp4(B, INTER, HIDDEN, E, TOPK, timing, num_iters, num_warmup):
    inter, w_down, w_scale, router_ids, router_wts, w_deq = _gen_down_fp4(
        B, INTER, HIDDEN, E, TOPK
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=inter.device)
    ref = _ref_down_fp4(inter, w_deq, router_ids, router_wts)  # not timed

    fn = lambda: flydsl_warp_decode_down_reduce_fp4(  # noqa: E731
        inter,
        w_down,
        router_ids,
        router_wts,
        w_scale,
        scale_block=(1, _MXFP4_BK),
        out=out,
    )
    got, us = run_perftest(
        fn,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=_rotate_for(inter, w_down, w_scale),
        **_timing_kwargs(timing),
    )
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        tol_err_ratio=0.1,
        msg="down fp4",
        printLog=False,
    )
    assert _cosine(ref, got) >= 0.99, "down fp4: correctness regression"
    # B3: single source of truth for the byte/FLOP model (weight_stream, FP4 weights
    # + E8M0 block-scale bytes at INTER/_MXFP4_BK per row).
    m = compute_metrics("down", B, HIDDEN, INTER, TOPK, "fp4", us)
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": m["TFLOPS"],
        "TB/s": m["TB/s"],
        "%peak": m["%peak"],
        "err": err,
    }


# -------------------------------------------------------------------------
# Cold-HBM E=256 harness (SILOTIGER-667 ?2): the FP4 BW win only shows when the
# weight pool >> MALL and successive decodes route to *different* experts, so a
# steady-state launch actually streams weights from HBM.  At E=8 the touched
# weights (TOPK experts) fit in MALL and stay warm; here we allocate the full
# E-expert pool once and rotate a tiny router-id set across disjoint expert
# groups so each launch reads a fresh slice of the (multi-GB) pool.
# -------------------------------------------------------------------------
# B, INTER, HIDDEN, E, TOPK.  DeepSeek-V3 down at the **real E=256** (FP4 pool
# ~1.88 GB, >> the ~256 MB MALL, so router-rotated reads are cold).  After the K3
# Tier-1 offset restructure (w_row*(DIM//WPACK) + k_base//WPACK) FP4 addresses up
# to E*HIDDEN*INTER < 2^32; FP8's byte offset is 2x larger, so its E=256 leg
# (3.76 GB, needs the K3 Tier-2 per-expert i64 base) is skipped and reported n/a.
# Real-E decode models for the widened cold-HBM sweep (Phase F, ?8.2 coverage
# gate).  Each model contributes both a down (B,INTER,HIDDEN,E,TOPK) and a
# gate_up (B,HIDDEN,INTER,E,TOPK) row across the full batch axis B?{1,2,4,8,32}.
# FP8 legs auto-skip where E*H*I >= 2^31 (DeepSeek E=256); FP4 measures for all.
# TP2/TP4 Qwen (INTER%512!=0) and Kimi-K3 (follow-on) are intentionally excluded.
# name, HIDDEN, INTER, E, TOPK.
_COLD_MODELS = [
    ("deepseek_v3", 7168, 2048, 256, 8),
    ("minimax", 3072, 1536, 256, 8),
    ("qwen3next_tp1", 2048, 512, 512, 10),
]
_COLD_BATCHES = [1, 2, 4, 8, 32]
# B1: FP8 cold legs stream a Block2D<128,128> weight scale to mirror CK's
# `Block2D<128,128>` layout (same scale-byte traffic + per-block scale work).
_FP8_SCALE_BLOCK = (128, 128)
# Correctness gate validates only the first few tokens' outputs (per-token work is
# uniform); bounds the reference's dequant footprint under distinct-per-token
# routing, where full-B at large B would touch (and fp32-dequant) the whole pool.
_COS_CHK_TOKENS = 4
COLD_DOWN_SHAPES = [
    (B, INTER, HIDDEN, E, TOPK)
    for (_name, HIDDEN, INTER, E, TOPK) in _COLD_MODELS
    for B in _COLD_BATCHES
]
# i32 addressing limits (see K3 scope). FP4 still addresses the whole pool with an
# i32 byte offset (w_row*INTER/2), overflowing at E*H*I >= 2^32. FP8 now uses the
# K3 Tier-2 per-expert i64 base (B5), so E*H*I no longer bounds it; the residual
# i32 limit is the *in-expert* byte offset (HIDDEN*INTER bytes/expert) vs 2^31.
_COLD_FP4_LIMIT = 2**32
_COLD_FP8_LIMIT = 2**31

# Weight element bytes by weight dtype (FP4 is packed 2/byte -> 0.5).
_WBYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0}
# Activation element bytes by activation dtype (gate_up: bf16 act by default;
# the FP8-activation variant (B4/gate_fp8_d2 peer) uses 1).
_XBYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0}


def compute_metrics(
    op, B, HIDDEN, INTER, TOPK, w_dtype, us, method="weight_stream", act_dtype="bf16"
):
    """Derived perf metrics from raw time -- the single source of truth for both
    the FlyDSL and CK harnesses (compare.py imports this and feeds CK's raw
    us+dims through it too, so one formula covers both sides).

    Time is the only method-independent quantity; bytes/FLOPs are pure functions
    of (op, dims, dtype, method).  Two named methods:

      "weight_stream" (DEFAULT) -- FlyDSL's weight-bandwidth-bound decode metric:
        the weight bytes a launch streams for its B*TOPK experts (distinct experts
        per token, so weight bytes scale with B; + e8m0 scale bytes for FP4) and the
        core MAC FLOPs.  Reproduces the cold benches' recorded wbytes/TB-s exactly.

      "total_traffic" -- mirrors CK commit 62e30c9098: every operand touched per
        launch (activation + both weights + intermediate + router + output) for
        bytes, and the full epilogue (silu/scale + router mul) for FLOPs.  MUST be
        revisited if CK's gu_bytes/dn_bytes/gu_flops/dn_flops change.

    op: "down" | "gate_up"; w_dtype/act_dtype in {"fp4","fp8","bf16"}.
    Returns {"TFLOPS", "TB/s", "%peak"} (%peak is bandwidth vs _HBM_PEAK_TBS).
    """
    if not (us and us > 0) or us != us:  # None / 0 / NaN
        return {"TFLOPS": float("nan"), "TB/s": float("nan"), "%peak": float("nan")}
    we = _WBYTES[w_dtype]
    xe = _XBYTES[act_dtype]

    if method == "weight_stream":
        if op == "down":
            outputs = B * HIDDEN
            flops = 2 * INTER * TOPK * outputs
            wbytes = outputs * TOPK * INTER * we
            if w_dtype == "fp4":  # + e8m0 block-scale bytes (1 B / 32-elt block)
                wbytes += outputs * TOPK * (INTER / _MXFP4_BK)
        elif op == "gate_up":
            out_elems = B * TOPK * INTER
            flops = 2 * 2 * HIDDEN * out_elems  # gate+up matmuls
            wbytes = out_elems * HIDDEN * 2 * we  # two weight streams (gate + up)
            if w_dtype == "fp4":
                wbytes += B * TOPK * 2 * INTER * (HIDDEN / _MXFP4_BK)
        else:
            raise ValueError(f"unknown op {op!r}")
        bytes_ = wbytes
    elif method == "total_traffic":
        if op == "down":
            flops = 3 * B * HIDDEN * TOPK * INTER
            bytes_ = (
                B
                * HIDDEN
                * (
                    TOPK * INTER * 2  # intermediate (bf16)
                    + TOPK * INTER * we  # down weight
                    + 2 * TOPK * 4  # router ids + wts (i32/f32)
                    + 2  # y (bf16)
                )
            )
        elif op == "gate_up":
            flops = B * TOPK * INTER * (4 * HIDDEN + 5)
            bytes_ = (
                B
                * TOPK
                * INTER
                * (
                    HIDDEN * xe  # activation
                    + 2 * HIDDEN * we  # gate + up weights
                    + 2  # intermediate (bf16)
                )
            )
        else:
            raise ValueError(f"unknown op {op!r}")
    else:
        raise ValueError(f"unknown method {method!r}")

    tbs = bytes_ / us / 1e6
    return {
        "TFLOPS": flops / us / 1e6,
        "TB/s": tbs,
        "%peak": 100.0 * tbs / _HBM_PEAK_TBS,
    }


def _router_group_list(B, E, TOPK, device):
    """Cold-HBM router-id sets with **distinct experts per token**, byte-for-byte
    matching the CK harness (`rids[i] = i % E` over `bk = B*TOPK` per launch).

    Each launch reads `B*TOPK` distinct experts, so weight HBM traffic scales with
    B -- exactly what CK reads and what `compute_metrics(weight_stream)` counts
    (fixes the earlier shared-expert asymmetry where FlyDSL read only TOPK experts
    while CK read B*TOPK).  `rotate = ceil(E / (B*TOPK))` disjoint groups tile the
    E-expert pool and consecutive launches march through it, so every timed launch
    reads its weights cold from HBM.  When `B*TOPK >= E` one launch already sweeps
    the whole pool and `rotate` collapses to 1 (still cold: pool >> cache)."""
    bk = B * TOPK
    rotate = max(1, (E + bk - 1) // bk)
    rid_list = []
    for g in range(rotate):
        flat = (g * bk + torch.arange(bk, device=device)) % E
        rid_list.append(flat.to(torch.int32).view(B, TOPK).contiguous())
    return rid_list


def _dequant_down_expert_fp4(w_down_e, w_scale_e, lut):
    """Unpack + dequantize one expert's FP4 down weights on the fly (fp32
    [HIDDEN, INTER]); used only for the touched experts of the correctness gate,
    so the full E-pool fp32 (~15 GB at E=256) is never materialized."""
    lo = (w_down_e & 0x0F).to(torch.long)
    hi = (w_down_e >> 4).to(torch.long)
    HIDDEN, half = w_down_e.shape
    INTER = half * 2
    codes = torch.empty(HIDDEN, INTER, dtype=torch.long, device=w_down_e.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    blk = torch.pow(2.0, w_scale_e.float() - 127.0)
    return lut[codes] * blk.repeat_interleave(_MXFP4_BK, dim=1)


def _gen_down_fp4_pool(B, INTER, HIDDEN, E, TOPK):
    """Full E-expert MXFP4 down pool (packed uint8 weights + E8M0 scales) built
    directly in packed form (no per-element int32/fp32 pool), plus a rotating
    router-id list.  Returns (inter, w_down, w_scale, rid_list, rwt)."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    rows = E * HIDDEN
    scale_cols = INTER // _MXFP4_BK
    # Packed FP4 (2 nibbles/byte) straight to uint8 -> 0.5 B/elt, no int32 pool.
    w_down = torch.randint(
        0, 256, (E, HIDDEN, INTER // 2), generator=gen, device=device, dtype=torch.uint8
    ).contiguous()
    # E8M0 bytes in a moderate power-of-two range (2^-4..2^4) => exact dequant.
    w_scale = torch.randint(
        123, 132, (rows, scale_cols), generator=gen, device=device, dtype=torch.uint8
    ).contiguous()
    inter = ((torch.rand((B, TOPK, INTER), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    rwt = torch.rand((B, TOPK), generator=gen, device=device).float()
    rwt = rwt / rwt.sum(dim=1, keepdim=True)
    rid_list = _router_group_list(B, E, TOPK, device)
    return inter, w_down, w_scale, rid_list, rwt


def _ref_down_fp4_pool(inter, w_down, w_scale, rid, rwt, HIDDEN):
    B, TOPK, INTER = inter.shape
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=inter.device)
    interf = inter.float()
    y = torch.zeros(B, HIDDEN, device=inter.device)
    cache = {}
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            if e not in cache:
                sc = w_scale[e * HIDDEN : (e + 1) * HIDDEN]
                cache[e] = _dequant_down_expert_fp4(w_down[e], sc, lut)
            y[b] += float(rwt[b, k]) * (interf[b, k] @ cache[e].T)
    return y.to(torch.bfloat16)


def _gen_down_fp8_pool(B, INTER, HIDDEN, E, TOPK, scale_block=_FP8_SCALE_BLOCK):
    """Full E-expert FP8 (e4m3) down pool + Block2D weight scale + rotating router
    list, matching :func:`_gen_down_fp4_pool` (same inter/rwt RNG stream).

    B1: the scale layout mirrors CK's ``Block2D<128,128>`` so the FP8 leg streams
    the same weight-scale bytes and does the same per-block scale work as CK."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    w_down = (
        ((torch.rand((E, HIDDEN, INTER), generator=gen, device=device) * 2 - 1) * 0.25)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    bn, bk = scale_block
    n_scale = _n_block2d_scales(E * HIDDEN, INTER, bn, bk)
    w_scale = (torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5).float()
    inter = ((torch.rand((B, TOPK, INTER), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    rwt = torch.rand((B, TOPK), generator=gen, device=device).float()
    rwt = rwt / rwt.sum(dim=1, keepdim=True)
    rid_list = _router_group_list(B, E, TOPK, device)
    return inter, w_down, w_scale, rid_list, rwt


def _ref_down_fp8_pool(
    inter, w_down, w_scale, rid, rwt, HIDDEN, scale_block=_FP8_SCALE_BLOCK
):
    B, TOPK, INTER = inter.shape
    bn, bk = scale_block
    interf = inter.float()
    y = torch.zeros(B, HIDDEN, device=inter.device)
    cache = {}
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            if e not in cache:
                sd = _block2d_scale_matrix(w_scale, e, HIDDEN, INTER, bn, bk)
                cache[e] = w_down[e].float() * sd
            y[b] += float(rwt[b, k]) * (interf[b, k] @ cache[e].T)
    return y.to(torch.bfloat16)


def _time_rotating(entry_fn, rid_list, num_iters, num_warmup, timing):
    """Time `entry_fn(router_ids)` while cycling `rid_list` across launches so
    steady-state reads sweep the whole cold pool (weights are captured by the
    closure -> not deep-copied, so no OOM)."""
    state = {"i": 0}

    def fn():
        rid = rid_list[state["i"] % len(rid_list)]
        state["i"] += 1
        return entry_fn(rid)

    got, us = run_perftest(
        fn,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,  # rotate content via the closure, not arg deep-copies
        **_timing_kwargs(timing),
    )
    return got, us


@benchmark()
def bench_down_cold(B, INTER, HIDDEN, E, TOPK, timing, num_iters, num_warmup):
    """Cold-HBM A/B: FP4 vs FP8 `down` at real E, router rotated over the pool.

    Returns one merged row (FP4 + FP8 side by side).  Metrics come from
    ``compute_metrics(method="weight_stream")``, which counts the weights a single
    launch reads -- B*TOPK experts' rows (distinct experts per token, matching CK)
    -- so TB/s reflects real HBM bandwidth and scales with B.
    """
    device = torch.device("cuda")
    nchk = min(B, _COS_CHK_TOKENS)
    ehi = E * HIDDEN * INTER
    # K3 addressing: FP4 still uses whole-pool i32 (byte offset w_row*INTER/2 -> caps
    # at E*H*I < 2^32). FP8 now uses the K3 Tier-2 per-expert i64 base (B5), so the
    # only residual i32 constraint is the *in-expert* offset (H*I bytes/expert).
    assert (
        ehi < _COLD_FP4_LIMIT
    ), f"E*HIDDEN*INTER={ehi} overflows even FP4's i32 offset (needs K3 Tier-2)"
    run_fp8 = HIDDEN * INTER < _COLD_FP8_LIMIT

    # ---- FP4 ----
    inter, w_down, w_scale, rid_list, rwt = _gen_down_fp4_pool(
        B, INTER, HIDDEN, E, TOPK
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=device)
    entry4 = lambda rid, inter=inter, w_down=w_down, w_scale=w_scale, out=out: flydsl_warp_decode_down_reduce_fp4(  # noqa: E731
        inter, w_down, rid, rwt, w_scale, scale_block=(1, _MXFP4_BK), out=out
    )
    got4 = entry4(rid_list[0])
    torch.cuda.synchronize()
    ref4 = _ref_down_fp4_pool(
        inter[:nchk], w_down, w_scale, rid_list[0][:nchk], rwt[:nchk], HIDDEN
    )
    cos4 = _cosine(ref4, got4[:nchk])
    assert cos4 >= 0.99, f"down fp4 cold: correctness regression (cos={cos4:.4f})"
    _, us4 = _time_rotating(entry4, rid_list, num_iters, num_warmup, timing)
    m4 = compute_metrics("down", B, HIDDEN, INTER, TOPK, "fp4", us4)
    tbs4 = m4["TB/s"]
    del inter, w_down, w_scale, out
    torch.cuda.empty_cache()

    # ---- FP8 (Block2D<128,128>, B1) -- K3 Tier-2 i64 base carries E*H*I >= 2^31 ----
    if run_fp8:
        inter8, w_down8, w_scale8, rid_list8, rwt8 = _gen_down_fp8_pool(
            B, INTER, HIDDEN, E, TOPK
        )
        out8 = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=device)
        entry8 = lambda rid, inter8=inter8, w_down8=w_down8, w_scale8=w_scale8, out8=out8: flydsl_warp_decode_down_reduce(  # noqa: E731
            inter8,
            w_down8,
            rid,
            rwt8,
            w_scale8,
            w_scale_mode="block2d",
            scale_block=_FP8_SCALE_BLOCK,
            out=out8,
        )
        got8 = entry8(rid_list8[0])
        torch.cuda.synchronize()
        ref8 = _ref_down_fp8_pool(
            inter8[:nchk], w_down8, w_scale8, rid_list8[0][:nchk], rwt8[:nchk], HIDDEN
        )
        cos8 = _cosine(ref8, got8[:nchk])
        assert cos8 >= 0.99, f"down fp8 cold: correctness regression (cos={cos8:.4f})"
        _, us8 = _time_rotating(entry8, rid_list8, num_iters, num_warmup, timing)
        tbs8 = compute_metrics("down", B, HIDDEN, INTER, TOPK, "fp8", us8)["TB/s"]
        del inter8, w_down8, w_scale8, out8
        torch.cuda.empty_cache()
    else:
        us8 = tbs8 = cos8 = float(
            "nan"
        )  # per-expert H*I offset would overflow i32 (beyond K3 Tier-2)

    return {
        "gfx": get_gfx(),
        "B": B,
        "INTER": INTER,
        "HIDDEN": HIDDEN,
        "E": E,
        "TOPK": TOPK,
        "fp4_us": us4,
        "fp8_us": us8,
        "fp4/fp8": us4 / us8 if us8 > 0 else float("nan"),
        "fp4_TB/s": tbs4,
        "fp8_TB/s": tbs8,
        "fp4_cos": cos4,
        "fp8_cos": cos8,
        "TFLOPS_fp4": m4["TFLOPS"],
    }


# -------------------------------------------------------------------------
# Cold-HBM gate_up A/B (mirror of bench_down_cold).  gate_up's contraction is
# HIDDEN and it streams *two* weight matrices (gate + up), so the per-launch cold
# read is 2x a same-shape down; the grid (B*TOPK*INTER waves) is also far larger,
# so this measures whether FP4's byte-halving still wins when the stage is
# occupancy-bound rather than latency-bound (cf. the G7 finding).
# -------------------------------------------------------------------------
# B, HIDDEN, INTER, E, TOPK.  DeepSeek-V3 gate_up at the real E=256 (FP4 gate+up
# pool ~3.76 GB >> MALL).  FP4 addresses up to E*INTER*HIDDEN < 2^32; FP8's byte
# offset is 2x, so its E=256 leg is skipped (needs K3 Tier-2) and reported n/a.
COLD_GATE_UP_SHAPES = [
    (B, HIDDEN, INTER, E, TOPK)
    for (_name, HIDDEN, INTER, E, TOPK) in _COLD_MODELS
    for B in _COLD_BATCHES
]


def _dequant_expert_fp4_rows(w_e, s_e, lut):
    """Unpack + dequantize one expert's FP4 weight matrix (fp32 [ROWS, K]) given
    packed [ROWS, K//2] uint8 nibbles and [ROWS, K//_MXFP4_BK] E8M0 scale bytes.
    Generic over the contraction dim so it serves gate_up ([INTER, HIDDEN])."""
    lo = (w_e & 0x0F).to(torch.long)
    hi = (w_e >> 4).to(torch.long)
    rows, half = w_e.shape
    K = half * 2
    codes = torch.empty(rows, K, dtype=torch.long, device=w_e.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    blk = torch.pow(2.0, s_e.float() - 127.0)
    return lut[codes] * blk.repeat_interleave(_MXFP4_BK, dim=1)


def _gen_gate_up_fp4_pool(B, HIDDEN, INTER, E, TOPK):
    """Full E-expert MXFP4 gate_up pool (packed gate+up uint8 + E8M0 scales) +
    rotating router list.  Returns (x, wg, wgs, wu, wus, rid_list)."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    rows = E * INTER
    scale_cols = HIDDEN // _MXFP4_BK

    def _one():
        w = torch.randint(
            0,
            256,
            (E, INTER, HIDDEN // 2),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        ).contiguous()
        s = torch.randint(
            123,
            132,
            (rows, scale_cols),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        ).contiguous()
        return w, s

    wg, wgs = _one()
    wu, wus = _one()
    x = ((torch.rand((B, HIDDEN), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    rid_list = _router_group_list(B, E, TOPK, device)
    return x, wg, wgs, wu, wus, rid_list


def _ref_gate_up_fp4_pool(x, wg, wgs, wu, wus, rid, INTER, HIDDEN):
    B, _ = x.shape
    TOPK = rid.shape[1]
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=x.device)
    xf = x.float()
    out = torch.zeros(B, TOPK, INTER, device=x.device)
    cache = {}
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            if e not in cache:
                sr = slice(e * INTER, (e + 1) * INTER)
                gate_e = _dequant_expert_fp4_rows(wg[e], wgs[sr], lut)
                up_e = _dequant_expert_fp4_rows(wu[e], wus[sr], lut)
                cache[e] = (gate_e, up_e)
            gate_e, up_e = cache[e]
            gate = xf[b] @ gate_e.T
            up = xf[b] @ up_e.T
            out[b, k] = (gate * torch.sigmoid(gate)) * up
    return out.to(torch.bfloat16)


def _gen_gate_up_fp8_pool(B, HIDDEN, INTER, E, TOPK, scale_block=_FP8_SCALE_BLOCK):
    """Full E-expert FP8 gate_up pool + Block2D<128,128> weight scales + rotating
    router list.  B1: mirrors CK's ``Block2D<128,128>`` scale layout."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)

    def _one():
        return (
            (
                (torch.rand((E, INTER, HIDDEN), generator=gen, device=device) * 2 - 1)
                * 0.25
            )
            .to(torch.float8_e4m3fn)
            .contiguous()
        )

    bn, bk = scale_block
    n_scale = _n_block2d_scales(E * INTER, HIDDEN, bn, bk)

    def _scale():
        return (torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5).float()

    wg = _one()
    wu = _one()
    wgs = _scale()
    wus = _scale()
    x = ((torch.rand((B, HIDDEN), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    rid_list = _router_group_list(B, E, TOPK, device)
    return x, wg, wgs, wu, wus, rid_list


def _ref_gate_up_fp8_pool(
    x, wg, wgs, wu, wus, rid, INTER, HIDDEN, scale_block=_FP8_SCALE_BLOCK
):
    B, _ = x.shape
    TOPK = rid.shape[1]
    bn, bk = scale_block
    xf = x.float()
    out = torch.zeros(B, TOPK, INTER, device=x.device)
    cache = {}
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            if e not in cache:
                sg = _block2d_scale_matrix(wgs, e, INTER, HIDDEN, bn, bk)
                su = _block2d_scale_matrix(wus, e, INTER, HIDDEN, bn, bk)
                cache[e] = (wg[e].float() * sg, wu[e].float() * su)
            gate_e, up_e = cache[e]
            gate = xf[b] @ gate_e.T
            up = xf[b] @ up_e.T
            out[b, k] = (gate * torch.sigmoid(gate)) * up
    return out.to(torch.bfloat16)


_FP8_E4M3_MAX = 448.0


def _gen_gate_up_fp8act_pool(
    B, HIDDEN, INTER, E, TOPK, scale_block=_FP8_SCALE_BLOCK, x_bk=128
):
    """FP8-activation gate_up pool (B4 / CK ``gate_fp8_d2`` peer): FP8 e4m3 ``x``
    with a ``Block2D<1, x_bk>`` per-(token, K-block) scale, plus the same FP8
    weights + ``Block2D<128,128>`` weight scales as :func:`_gen_gate_up_fp8_pool`
    (identical RNG stream, so the weight legs stay comparable)."""
    x, wg, wgs, wu, wus, rid_list = _gen_gate_up_fp8_pool(
        B, HIDDEN, INTER, E, TOPK, scale_block=scale_block
    )
    # Quantize the bf16 x into FP8 with a per-(token, x_bk) block scale (amax/max).
    nblk = HIDDEN // x_bk
    xr = x.float().view(B, nblk, x_bk)
    amax = xr.abs().amax(dim=2).clamp(min=1e-8)  # [B, nblk]
    x_scale = (amax / _FP8_E4M3_MAX).float()  # [B, nblk]
    x_fp8 = (xr / x_scale[..., None]).view(B, HIDDEN).to(torch.float8_e4m3fn)
    x_scale_flat = x_scale.reshape(-1).contiguous()  # row-major (token, K-block)
    return x_fp8, x_scale_flat, wg, wgs, wu, wus, rid_list


def _ref_gate_up_fp8act_pool(
    x_fp8,
    x_scale,
    wg,
    wgs,
    wu,
    wus,
    rid,
    INTER,
    HIDDEN,
    scale_block=_FP8_SCALE_BLOCK,
    x_bk=128,
):
    B, _ = x_fp8.shape
    TOPK = rid.shape[1]
    bn, bk = scale_block
    nblk = HIDDEN // x_bk
    xs = x_scale.view(-1, nblk)[:B]  # tolerate a full-B x_scale with a sliced x
    xf = (x_fp8.float().view(B, nblk, x_bk) * xs[..., None]).view(B, HIDDEN)
    out = torch.zeros(B, TOPK, INTER, device=x_fp8.device)
    cache = {}
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            if e not in cache:
                sg = _block2d_scale_matrix(wgs, e, INTER, HIDDEN, bn, bk)
                su = _block2d_scale_matrix(wus, e, INTER, HIDDEN, bn, bk)
                cache[e] = (wg[e].float() * sg, wu[e].float() * su)
            gate_e, up_e = cache[e]
            gate = xf[b] @ gate_e.T
            up = xf[b] @ up_e.T
            out[b, k] = (gate * torch.sigmoid(gate)) * up
    return out.to(torch.bfloat16)


@benchmark()
def bench_gate_up_cold(B, HIDDEN, INTER, E, TOPK, timing, num_iters, num_warmup):
    """Cold-HBM A/B: FP4 vs FP8 `gate_up` at real E, router rotated over the pool.

    Metrics come from ``compute_metrics(method="weight_stream")``, which counts the
    two weight streams (gate + up) a single launch reads for its B*TOPK experts
    (distinct per token, matching CK), so TB/s reflects real HBM bandwidth.
    """
    device = torch.device("cuda")
    nchk = min(B, _COS_CHK_TOKENS)
    ehi = E * INTER * HIDDEN
    # FP4 stays on whole-pool i32 (caps at E*I*H < 2^32); FP8 uses the K3 Tier-2
    # per-expert i64 base (B5), so its only residual i32 limit is the in-expert
    # offset (INTER*HIDDEN bytes/expert).
    assert (
        ehi < _COLD_FP4_LIMIT
    ), f"E*INTER*HIDDEN={ehi} overflows even FP4's i32 offset (needs K3 Tier-2)"
    run_fp8 = INTER * HIDDEN < _COLD_FP8_LIMIT

    # ---- FP4 ----
    x, wg, wgs, wu, wus, rid_list = _gen_gate_up_fp4_pool(B, HIDDEN, INTER, E, TOPK)
    out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=device)
    entry4 = lambda rid, x=x, wg=wg, wgs=wgs, wu=wu, wus=wus, out=out: flydsl_warp_decode_gate_up_fp4(  # noqa: E731
        x, wg, wu, rid, wgs, wus, scale_block=(1, _MXFP4_BK), out=out
    )
    got4 = entry4(rid_list[0])
    torch.cuda.synchronize()
    ref4 = _ref_gate_up_fp4_pool(
        x[:nchk], wg, wgs, wu, wus, rid_list[0][:nchk], INTER, HIDDEN
    )
    cos4 = _cosine(ref4, got4[:nchk])
    assert cos4 >= 0.99, f"gate_up fp4 cold: correctness regression (cos={cos4:.4f})"
    _, us4 = _time_rotating(entry4, rid_list, num_iters, num_warmup, timing)
    m4 = compute_metrics("gate_up", B, HIDDEN, INTER, TOPK, "fp4", us4)
    tbs4 = m4["TB/s"]
    del x, wg, wgs, wu, wus, out
    torch.cuda.empty_cache()

    # ---- FP8 (Block2D<128,128>, B1) -- K3 Tier-2 i64 base carries E*I*H >= 2^31 ----
    if run_fp8:
        x8, wg8, wgs8, wu8, wus8, rid_list8 = _gen_gate_up_fp8_pool(
            B, HIDDEN, INTER, E, TOPK
        )
        out8 = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=device)
        entry8 = lambda rid, x8=x8, wg8=wg8, wu8=wu8, wgs8=wgs8, wus8=wus8, out8=out8: flydsl_warp_decode_gate_up(  # noqa: E731
            x8,
            wg8,
            wu8,
            rid,
            wgs8,
            wus8,
            w_scale_mode="block2d",
            scale_block=_FP8_SCALE_BLOCK,
            out=out8,
        )
        got8 = entry8(rid_list8[0])
        torch.cuda.synchronize()
        ref8 = _ref_gate_up_fp8_pool(
            x8[:nchk], wg8, wgs8, wu8, wus8, rid_list8[0][:nchk], INTER, HIDDEN
        )
        cos8 = _cosine(ref8, got8[:nchk])
        assert (
            cos8 >= 0.99
        ), f"gate_up fp8 cold: correctness regression (cos={cos8:.4f})"
        _, us8 = _time_rotating(entry8, rid_list8, num_iters, num_warmup, timing)
        tbs8 = compute_metrics("gate_up", B, HIDDEN, INTER, TOPK, "fp8", us8)["TB/s"]
        del x8, wg8, wu8, out8
        torch.cuda.empty_cache()
    else:
        us8 = tbs8 = cos8 = float(
            "nan"
        )  # per-expert I*H offset would overflow i32 (beyond K3 Tier-2)

    # ---- FP8-activation (B4, CK gate_fp8_d2 peer): FP8 x + Block2D<1,128> x-scale --
    if run_fp8:
        xa, xsa, wga, wgsa, wua, wusa, rid_lista = _gen_gate_up_fp8act_pool(
            B, HIDDEN, INTER, E, TOPK
        )
        outa = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=device)
        entrya = lambda rid, xa=xa, xsa=xsa, wga=wga, wgsa=wgsa, wua=wua, wusa=wusa, outa=outa: flydsl_warp_decode_gate_up_fp8act(  # noqa: E731
            xa,
            wga,
            wua,
            rid,
            xsa,
            wgsa,
            wusa,
            scale_block=_FP8_SCALE_BLOCK,
            out=outa,
        )
        gota = entrya(rid_lista[0])
        torch.cuda.synchronize()
        refa = _ref_gate_up_fp8act_pool(
            xa[:nchk], xsa, wga, wgsa, wua, wusa, rid_lista[0][:nchk], INTER, HIDDEN
        )
        cosa = _cosine(refa, gota[:nchk])
        assert (
            cosa >= 0.99
        ), f"gate_up fp8-act cold: correctness regression (cos={cosa:.4f})"
        _, usa = _time_rotating(entrya, rid_lista, num_iters, num_warmup, timing)
        tbsa = compute_metrics(
            "gate_up", B, HIDDEN, INTER, TOPK, "fp8", usa, act_dtype="fp8"
        )["TB/s"]
        del xa, wga, wua, outa
        torch.cuda.empty_cache()
    else:
        usa = tbsa = cosa = float("nan")

    return {
        "gfx": get_gfx(),
        "B": B,
        "HIDDEN": HIDDEN,
        "INTER": INTER,
        "E": E,
        "TOPK": TOPK,
        "fp4_us": us4,
        "fp8_us": us8,
        "fp8act_us": usa,
        "fp4/fp8": us4 / us8 if us8 > 0 else float("nan"),
        "fp4_TB/s": tbs4,
        "fp8_TB/s": tbs8,
        "fp8act_TB/s": tbsa,
        "fp4_cos": cos4,
        "fp8_cos": cos8,
        "fp8act_cos": cosa,
        "TFLOPS_fp4": m4["TFLOPS"],
    }


def _fmt_table(rows) -> str:
    """Markdown table when ``tabulate`` is available; plain text otherwise."""
    df = pd.DataFrame(rows)
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _run_perf_sweeps(args) -> None:
    """Correctness+perf sweeps -> one markdown table per stage."""
    timing_kw = dict(
        timing=args.timing, num_iters=args.num_iters, num_warmup=args.num_warmup
    )

    gate_rows = [
        bench_gate_up(B, HIDDEN, INTER, E, TOPK, mode, scale_block=sb, **timing_kw)
        for (B, HIDDEN, INTER, E, TOPK, mode, sb) in args.gate_up_shapes
    ]
    aiter.logger.info(
        "warp-decode gate_up perf (%s timing):\n%s", args.timing, _fmt_table(gate_rows)
    )

    down_rows = [
        bench_down(B, INTER, HIDDEN, E, TOPK, mode, scale_block=sb, **timing_kw)
        for (B, INTER, HIDDEN, E, TOPK, mode, sb) in args.down_shapes
    ]
    aiter.logger.info(
        "warp-decode down_reduce perf (%s timing):\n%s",
        args.timing,
        _fmt_table(down_rows),
    )

    down_fp4_rows = [
        bench_down_fp4(B, INTER, HIDDEN, E, TOPK, **timing_kw)
        for (B, INTER, HIDDEN, E, TOPK) in args.down_fp4_shapes
    ]
    aiter.logger.info(
        "warp-decode down_reduce MXFP4 perf (%s timing) "
        "[A/B vs FP8 at matching B/INTER/HIDDEN]:\n%s",
        args.timing,
        _fmt_table(down_fp4_rows),
    )

    if getattr(args, "cold", False):
        cold_rows = [
            bench_down_cold(B, INTER, HIDDEN, E, TOPK, **timing_kw)
            for (B, INTER, HIDDEN, E, TOPK) in args.cold_down_shapes
        ]
        aiter.logger.info(
            "warp-decode down_reduce COLD-HBM A/B FP4-vs-FP8 (real-E decode models "
            "x B in {1,2,4,8,32}; %s timing; FP8 n/a where E*H*I>=2^31; router "
            "rotated over the pool):\n%s",
            args.timing,
            _fmt_table(cold_rows),
        )
        cold_gu_rows = [
            bench_gate_up_cold(B, HIDDEN, INTER, E, TOPK, **timing_kw)
            for (B, HIDDEN, INTER, E, TOPK) in args.cold_gate_up_shapes
        ]
        aiter.logger.info(
            "warp-decode gate_up COLD-HBM A/B FP4-vs-FP8 (real-E decode models "
            "x B in {1,2,4,8,32}; %s timing; FP8 n/a where E*H*I>=2^31; router "
            "rotated over the pool):\n%s",
            args.timing,
            _fmt_table(cold_gu_rows),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-serialize",
        action="store_true",
        help="disable the s_nop 2 dot2 hazard guard",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="run only the correctness checks (no perf sweep / tables)",
    )
    parser.add_argument(
        "--timing",
        choices=["device", "cuda_event", "graph"],
        default="device",
        help="run_perftest timing mode for the perf sweep (default: device)",
    )
    parser.add_argument("--num-iters", type=int, default=_PERF_NUM_ITERS)
    parser.add_argument("--num-warmup", type=int, default=_PERF_NUM_WARMUP)
    parser.add_argument(
        "--cold",
        action="store_true",
        help="also run the cold-HBM E=256 FP4-vs-FP8 down A/B (allocates multi-GB "
        "weight pools; router-id rotated so steady-state reads miss MALL)",
    )
    args = parser.parse_args()
    # Fixed realistic shapes (weights >> LLC); not swept via CLI for now.
    args.gate_up_shapes = GATE_UP_PERF_SHAPES
    args.down_shapes = DOWN_PERF_SHAPES
    args.down_fp4_shapes = DOWN_FP4_PERF_SHAPES
    args.cold_down_shapes = COLD_DOWN_SHAPES
    args.cold_gate_up_shapes = COLD_GATE_UP_SHAPES

    print("=" * 78)
    print("[flydsl] warp-decode MoE primitives (Phase 1)")
    print("=" * 78)
    inputs, outputs = _run_primitives(serialize_dot2=not args.no_serialize)

    results = [
        _check_dot2(inputs, outputs),
        _check_convert(inputs, outputs) if _HAS_FP8 else (True, 0.0),
        _check_reduce(inputs, outputs),
    ]
    n_pass = sum(1 for p, _ in results if p)
    print(f"\n  {n_pass}/{len(results)} primitive checks passed")

    print("\n" + "=" * 78)
    print("[flydsl] warp-decode MoE gate_up FP8 (Phase 2)")
    print("=" * 78)
    gate_up_ok = True
    if _HAS_FP8:
        for case in GATE_UP_CASES:
            passed, _ = _run_gate_up_case(case)
            gate_up_ok = gate_up_ok and passed
    else:
        print("  skipped (torch build lacks float8_e4m3fn)")

    print("\n" + "=" * 78)
    print("[flydsl] warp-decode MoE down_reduce FP8 (Phase 3)")
    print("=" * 78)
    down_ok = True
    if _HAS_FP8:
        for case in DOWN_CASES:
            passed, _ = _run_down_case(case)
            down_ok = down_ok and passed
    else:
        print("  skipped (torch build lacks float8_e4m3fn)")

    all_ok = (n_pass == len(results)) and gate_up_ok and down_ok

    if not args.skip_perf and _HAS_FP8:
        print("\n" + "=" * 78)
        print(f"[flydsl] warp-decode MoE perf sweep (timing={args.timing})")
        print("=" * 78)
        _run_perf_sweeps(args)
    elif not _HAS_FP8:
        print("\n  perf sweep skipped (torch build lacks float8_e4m3fn)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
