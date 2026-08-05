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

# name, B, HIDDEN, INTER, E, TOPK, w_scale_mode
GATE_UP_CASES = [
    ("h1024_i64_e4_tk2_pertensor", 2, 1024, 64, 4, 2, "pertensor"),
    ("h1024_i128_e8_tk2_pertoken", 1, 1024, 128, 8, 2, "pertoken"),
    ("h512_i32_e2_tk1_kv8_pertensor", 1, 512, 32, 2, 1, "pertensor"),
]


def _gen_gate_up(B, HIDDEN, INTER, E, TOPK, w_scale_mode):
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
    n_scale = 1 if w_scale_mode == "pertensor" else E * INTER
    w_gate_scale = (
        torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5
    ).float()
    w_up_scale = (torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5).float()
    return x, w_gate, w_up, router_ids, w_gate_scale, w_up_scale


def _ref_gate_up(x, w_gate, w_up, router_ids, w_gate_scale, w_up_scale, w_scale_mode):
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
    name, B, HIDDEN, INTER, E, TOPK, mode = case
    print("=" * 78)
    print(f"[flydsl] warp-decode gate_up  case={name}")
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        B, HIDDEN, INTER, E, TOPK, mode
    )
    out = flydsl_warp_decode_gate_up(
        x, w_gate, w_up, router_ids, wgs, wus, w_scale_mode=mode
    )
    torch.cuda.synchronize()
    ref = _ref_gate_up(x, w_gate, w_up, router_ids, wgs, wus, mode)
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
# name, B, INTER, HIDDEN, E, TOPK, w_scale_mode
DOWN_CASES = [
    ("down_i1024_h64_e4_tk2_pertensor", 2, 1024, 64, 4, 2, "pertensor"),
    ("down_i1024_h128_e8_tk2_pertoken", 1, 1024, 128, 8, 2, "pertoken"),
    ("down_i512_h32_e2_tk1_kv8_pertensor", 1, 512, 32, 2, 1, "pertensor"),
]


def _gen_down(B, INTER, HIDDEN, E, TOPK, w_scale_mode):
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
    n_scale = 1 if w_scale_mode == "pertensor" else E * HIDDEN
    w_down_scale = (
        torch.rand(n_scale, generator=gen, device=device) * 1.5 + 0.5
    ).float()
    return inter, w_down, router_ids, router_wts, w_down_scale


def _ref_down(inter, w_down, router_ids, router_wts, w_down_scale, w_scale_mode):
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
            dot = interf[b, k] @ wdf[e].T
            if w_scale_mode == "pertensor":
                ds = w_down_scale[0]
            else:
                ds = w_down_scale[e * HIDDEN + idx]
            y[b] += dot * (rw * ds)
    return y.to(torch.bfloat16)


def _run_down_case(case, *, cos_thresh=0.999):
    name, B, INTER, HIDDEN, E, TOPK, mode = case
    print("=" * 78)
    print(f"[flydsl] warp-decode down_reduce  case={name}")
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        B, INTER, HIDDEN, E, TOPK, mode
    )
    out = flydsl_warp_decode_down_reduce(
        inter, w_down, router_ids, router_wts, wds, w_scale_mode=mode
    )
    torch.cuda.synchronize()
    ref = _ref_down(inter, w_down, router_ids, router_wts, wds, mode)
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
# Perf sweep -- combined correctness + benchmark (SILOTIGER-667 ?2 standard)
# -------------------------------------------------------------------------
# Realistic decode shapes (weights >> last-level cache). name, B, HIDDEN, INTER,
# E, TOPK, mode.  DeepSeek-V3-ish (H7168/I2048/TOPK8) is the headline case.
GATE_UP_PERF_SHAPES = [
    (1, 7168, 2048, 8, 8, "pertensor"),
    (4, 7168, 2048, 8, 8, "pertensor"),
    (1, 7168, 2048, 8, 8, "pertoken"),
    (1, 4096, 1024, 8, 8, "pertensor"),
]
# name, B, INTER, HIDDEN, E, TOPK, mode.
DOWN_PERF_SHAPES = [
    (1, 2048, 7168, 8, 8, "pertensor"),
    (4, 2048, 7168, 8, 8, "pertensor"),
    (1, 2048, 7168, 8, 8, "pertoken"),
    (1, 1024, 4096, 8, 8, "pertensor"),
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
def bench_gate_up(B, HIDDEN, INTER, E, TOPK, mode, timing, num_iters, num_warmup):
    x, w_gate, w_up, router_ids, wgs, wus = _gen_gate_up(
        B, HIDDEN, INTER, E, TOPK, mode
    )
    # Faithful to the real call: pre-allocate and pass the output buffer.
    out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=x.device)
    ref = _ref_gate_up(x, w_gate, w_up, router_ids, wgs, wus, mode)  # not timed

    outputs = B * TOPK * INTER
    flops = 4 * HIDDEN * outputs  # 2 dots (gate+up) x 2 (mul+add) x HIDDEN
    wbytes = 2 * outputs * HIDDEN  # gate+up FP8 rows streamed (1 B each)

    fn = lambda: flydsl_warp_decode_gate_up(  # noqa: E731
        x, w_gate, w_up, router_ids, wgs, wus, w_scale_mode=mode, out=out
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
def bench_down(B, INTER, HIDDEN, E, TOPK, mode, timing, num_iters, num_warmup):
    inter, w_down, router_ids, router_wts, wds = _gen_down(
        B, INTER, HIDDEN, E, TOPK, mode
    )
    out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=inter.device)
    ref = _ref_down(inter, w_down, router_ids, router_wts, wds, mode)  # not timed

    outputs = B * HIDDEN
    flops = 2 * INTER * TOPK * outputs  # TOPK dots of length INTER x 2 (mul+add)
    wbytes = outputs * TOPK * INTER  # FP8 down rows streamed (1 B each)

    fn = lambda: flydsl_warp_decode_down_reduce(  # noqa: E731
        inter, w_down, router_ids, router_wts, wds, w_scale_mode=mode, out=out
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
        bench_gate_up(B, HIDDEN, INTER, E, TOPK, mode, **timing_kw)
        for (B, HIDDEN, INTER, E, TOPK, mode) in args.gate_up_shapes
    ]
    aiter.logger.info(
        "warp-decode gate_up perf (%s timing):\n%s", args.timing, _fmt_table(gate_rows)
    )

    down_rows = [
        bench_down(B, INTER, HIDDEN, E, TOPK, mode, **timing_kw)
        for (B, INTER, HIDDEN, E, TOPK, mode) in args.down_shapes
    ]
    aiter.logger.info(
        "warp-decode down_reduce perf (%s timing):\n%s",
        args.timing,
        _fmt_table(down_rows),
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
    args = parser.parse_args()
    # Fixed realistic shapes (weights >> LLC); not swept via CLI for now.
    args.gate_up_shapes = GATE_UP_PERF_SHAPES
    args.down_shapes = DOWN_PERF_SHAPES

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
