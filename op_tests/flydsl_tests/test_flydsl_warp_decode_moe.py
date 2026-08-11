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

    outputs = B * TOPK * INTER
    flops = 4 * HIDDEN * outputs  # 2 dots (gate+up) x 2 (mul+add) x HIDDEN
    wbytes = 2 * outputs * HIDDEN  # gate+up FP8 rows streamed (1 B each)

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
    tbs = wbytes / us / 1e6 if us > 0 else 0.0
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6 if us > 0 else 0.0,
        "TB/s": tbs,
        "%peak": 100.0 * tbs / _HBM_PEAK_TBS,
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

    outputs = B * HIDDEN
    flops = 2 * INTER * TOPK * outputs  # TOPK dots of length INTER x 2 (mul+add)
    wbytes = outputs * TOPK * INTER  # FP8 down rows streamed (1 B each)

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
    tbs = wbytes / us / 1e6 if us > 0 else 0.0
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6 if us > 0 else 0.0,
        "TB/s": tbs,
        "%peak": 100.0 * tbs / _HBM_PEAK_TBS,
        "err": err,
    }


@benchmark()
def bench_down_fp4(B, INTER, HIDDEN, E, TOPK, timing, num_iters, num_warmup):
    inter, w_down, w_scale, router_ids, router_wts, w_deq = _gen_down_fp4(
        B, INTER, HIDDEN, E, TOPK
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=inter.device)
    ref = _ref_down_fp4(inter, w_deq, router_ids, router_wts)  # not timed

    outputs = B * HIDDEN
    flops = 2 * INTER * TOPK * outputs  # identical to FP8: TOPK dots x 2 (mul+add)
    # FP4 down streams 0.5 B/elt of weights + the E8M0 scale bytes (INTER/32/row).
    wbytes = outputs * TOPK * INTER // 2 + outputs * TOPK * (INTER // _MXFP4_BK)

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
    tbs = wbytes / us / 1e6 if us > 0 else 0.0
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6 if us > 0 else 0.0,
        "TB/s": tbs,
        "%peak": 100.0 * tbs / _HBM_PEAK_TBS,
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
COLD_DOWN_SHAPES = [
    (1, 2048, 7168, 256, 8),
    (2, 2048, 7168, 256, 8),
    (4, 2048, 7168, 256, 8),
    (8, 2048, 7168, 256, 8),
]
# i32 addressing limits on E*HIDDEN*INTER (see K3 scope): FP4's hardware byte
# offset is w_row*INTER/2 (dword*4), so it overflows at E*H*I >= 2^32; FP8's is
# w_row*INTER (dword*4), overflowing at 2^31.
_COLD_FP4_LIMIT = 2**32
_COLD_FP8_LIMIT = 2**31


def _router_group_list(B, E, TOPK, device, n_route):
    """`n_route` router-id sets, each selecting a *disjoint* contiguous block of
    TOPK experts (group g -> experts [g*TOPK, (g+1)*TOPK)), wrapping mod E.  All
    B tokens in a set share the group so the union of reads across the list
    sweeps the whole pool (n_route*TOPK experts)."""
    rid_list = []
    for g in range(n_route):
        base = (g * TOPK) % E
        experts = [(base + j) % E for j in range(TOPK)]
        rid = torch.tensor(experts, dtype=torch.int32, device=device)
        rid_list.append(rid.view(1, TOPK).expand(B, TOPK).contiguous())
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


def _gen_down_fp4_pool(B, INTER, HIDDEN, E, TOPK, n_route):
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
    rid_list = _router_group_list(B, E, TOPK, device, n_route)
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


def _gen_down_fp8_pool(B, INTER, HIDDEN, E, TOPK, n_route):
    """Full E-expert FP8 (e4m3) down pool + PerTensor scale + rotating router
    list, matching :func:`_gen_down_fp4_pool` (same inter/rwt RNG stream)."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260811)
    w_down = (
        ((torch.rand((E, HIDDEN, INTER), generator=gen, device=device) * 2 - 1) * 0.25)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    w_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    inter = ((torch.rand((B, TOPK, INTER), generator=gen, device=device) * 2 - 1)).to(
        torch.bfloat16
    )
    rwt = torch.rand((B, TOPK), generator=gen, device=device).float()
    rwt = rwt / rwt.sum(dim=1, keepdim=True)
    rid_list = _router_group_list(B, E, TOPK, device, n_route)
    return inter, w_down, w_scale, rid_list, rwt


def _ref_down_fp8_pool(inter, w_down, w_scale, rid, rwt, HIDDEN):
    B, TOPK, _ = inter.shape
    interf = inter.float()
    scale = float(w_scale[0])
    y = torch.zeros(B, HIDDEN, device=inter.device)
    for b in range(B):
        for k in range(TOPK):
            e = int(rid[b, k])
            y[b] += float(rwt[b, k]) * (interf[b, k] @ w_down[e].float().T) * scale
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

    Returns one merged row (FP4 + FP8 side by side).  ``wbytes`` counts only the
    weights a single launch reads (TOPK experts' rows) -- the effective cold read
    per launch -- so TB/s reflects real HBM bandwidth, not the whole pool.
    """
    device = torch.device("cuda")
    n_route = max(1, E // TOPK)
    outputs = B * HIDDEN
    flops = 2 * INTER * TOPK * outputs
    ehi = E * HIDDEN * INTER
    # K3 i32 addressing limits (see scope): FP4 byte offset = w_row*INTER/2, FP8 =
    # w_row*INTER; the FP8 leg is skipped above 2^31 (needs the Tier-2 i64 base).
    assert (
        ehi < _COLD_FP4_LIMIT
    ), f"E*HIDDEN*INTER={ehi} overflows even FP4's i32 offset (needs K3 Tier-2)"
    run_fp8 = ehi < _COLD_FP8_LIMIT

    # ---- FP4 ----
    inter, w_down, w_scale, rid_list, rwt = _gen_down_fp4_pool(
        B, INTER, HIDDEN, E, TOPK, n_route
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=device)
    ref4 = _ref_down_fp4_pool(inter, w_down, w_scale, rid_list[0], rwt, HIDDEN)
    entry4 = lambda rid, inter=inter, w_down=w_down, w_scale=w_scale, out=out: flydsl_warp_decode_down_reduce_fp4(  # noqa: E731
        inter, w_down, rid, rwt, w_scale, scale_block=(1, _MXFP4_BK), out=out
    )
    got4 = entry4(rid_list[0])
    torch.cuda.synchronize()
    cos4 = _cosine(ref4, got4)
    assert cos4 >= 0.99, f"down fp4 cold: correctness regression (cos={cos4:.4f})"
    _, us4 = _time_rotating(entry4, rid_list, num_iters, num_warmup, timing)
    wbytes4 = outputs * TOPK * INTER // 2 + outputs * TOPK * (INTER // _MXFP4_BK)
    tbs4 = wbytes4 / us4 / 1e6 if us4 > 0 else 0.0
    del inter, w_down, w_scale, out
    torch.cuda.empty_cache()

    # ---- FP8 (PerTensor) -- only when the i32 byte offset fits (E*H*I < 2^31) ----
    if run_fp8:
        inter8, w_down8, w_scale8, rid_list8, rwt8 = _gen_down_fp8_pool(
            B, INTER, HIDDEN, E, TOPK, n_route
        )
        out8 = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=device)
        ref8 = _ref_down_fp8_pool(inter8, w_down8, w_scale8, rid_list8[0], rwt8, HIDDEN)
        entry8 = lambda rid, inter8=inter8, w_down8=w_down8, w_scale8=w_scale8, out8=out8: flydsl_warp_decode_down_reduce(  # noqa: E731
            inter8, w_down8, rid, rwt8, w_scale8, w_scale_mode="pertensor", out=out8
        )
        got8 = entry8(rid_list8[0])
        torch.cuda.synchronize()
        cos8 = _cosine(ref8, got8)
        assert cos8 >= 0.99, f"down fp8 cold: correctness regression (cos={cos8:.4f})"
        _, us8 = _time_rotating(entry8, rid_list8, num_iters, num_warmup, timing)
        wbytes8 = outputs * TOPK * INTER  # 1 B/elt FP8 weights
        tbs8 = wbytes8 / us8 / 1e6 if us8 > 0 else 0.0
        del inter8, w_down8, w_scale8, out8
        torch.cuda.empty_cache()
    else:
        us8 = tbs8 = float("nan")  # FP8 E=256 needs K3 Tier-2 per-expert i64 base

    return {
        "gfx": get_gfx(),
        "B": B,
        "E": E,
        "fp4_us": us4,
        "fp8_us": us8,
        "fp4/fp8": us4 / us8 if us8 > 0 else float("nan"),
        "fp4_TB/s": tbs4,
        "fp8_TB/s": tbs8,
        "fp4_cos": cos4,
        "TFLOPS_fp4": flops / us4 / 1e6 if us4 > 0 else 0.0,
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
            "warp-decode down_reduce COLD-HBM E=%d A/B FP4-vs-FP8 (%s timing, "
            "router rotated over the pool):\n%s",
            args.cold_down_shapes[0][3],
            args.timing,
            _fmt_table(cold_rows),
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
