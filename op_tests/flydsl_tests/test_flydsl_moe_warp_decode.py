# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for the FlyDSL warp-decode gate_up kernel.

SILOTIGER-667: warp-decode MoE MLP kernels for decode batch sizes B=1..4.

Usage
-----
    pytest op_tests/flydsl_tests/test_flydsl_moe_warp_decode.py -v
    pytest op_tests/flydsl_tests/test_flydsl_moe_warp_decode.py -v -k deepseek
    # Without pytest (direct run):
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python op_tests/flydsl_tests/test_flydsl_moe_warp_decode.py

Architecture gates
------------------
    use_dot2=False  FP32 scalar path — runs on gfx942 and gfx950.
    use_dot2=True   v_dot2_f32_bf16  — gfx950 only (skipped on gfx942).
"""

from __future__ import annotations

import time
from typing import List, Tuple

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

# Import the kernel directly to avoid pulling the full aiter package
# (which requires triton, pandas, etc.).  In CI the full aiter env is present.
try:
    from aiter.ops.flydsl.kernels.moe_warp_decode import compile_wd_moe_gate_up
except ImportError:
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location(
        "moe_warp_decode",
        pathlib.Path(__file__).parents[2]
        / "aiter/ops/flydsl/kernels/moe_warp_decode.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    compile_wd_moe_gate_up = _mod.compile_wd_moe_gate_up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rocm_arch() -> str:
    try:
        props = torch.cuda.get_device_properties(0)
        return props.gcnArchName.split(":")[0].lower()
    except Exception:
        return "unknown"


def _is_gfx950() -> bool:
    return _rocm_arch().startswith("gfx950") or _rocm_arch().startswith("gfx95")


def _ptr(t: torch.Tensor):
    """Convert a contiguous GPU tensor to a raw FlyDSL pointer."""
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _ref_gate_up(
    x: torch.Tensor,  # [B, hidden] bf16
    w_gate: torch.Tensor,  # [E*inter, hidden] bf16
    w_up: torch.Tensor,  # [E*inter, hidden] bf16
    router_ids: torch.Tensor,  # [B*topk] i32
    B: int,
    topk: int,
    inter: int,
) -> torch.Tensor:
    """CPU/GPU reference: silu(gate @ x) * (up @ x) for each (token, slot)."""
    ref = torch.zeros(B * topk, inter, dtype=torch.float32, device=x.device)
    xf = x.float()
    wgf = w_gate.float()
    wuf = w_up.float()
    for slot in range(B * topk):
        tok = slot // topk
        e = router_ids[slot].item()
        gv = wgf[e * inter : (e + 1) * inter] @ xf[tok]
        uv = wuf[e * inter : (e + 1) * inter] @ xf[tok]
        ref[slot] = torch.sigmoid(gv) * gv * uv
    return ref.to(torch.bfloat16)


def _run_kernel(
    exe,
    inter_out,
    x,
    w_gate,
    w_up,
    router_ids,
    B,
    topk,
    inter,
    hidden,
    experts,
    w_scale: float = 1.0,
):
    stream = torch.cuda.current_stream()
    exe(
        _ptr(inter_out),
        _ptr(x),
        _ptr(w_gate),
        _ptr(w_up),
        _ptr(router_ids),
        B,
        topk,
        inter,
        hidden,
        experts,
        w_scale,
        stream,
    )
    torch.cuda.synchronize()


def _ref_gate_up_fp8(
    x: torch.Tensor,  # [B, hidden] bf16
    w_gate_f: torch.Tensor,  # [E*inter, hidden] float32 (dequantised weights)
    w_up_f: torch.Tensor,
    router_ids: torch.Tensor,
    B: int,
    topk: int,
    inter: int,
) -> torch.Tensor:
    """Reference for BF16-act × FP8-weight path (FP8 roundtripped through float32)."""
    ref = torch.zeros(B * topk, inter, dtype=torch.float32, device=x.device)
    xf = x.float()
    for slot in range(B * topk):
        tok = slot // topk
        e = router_ids[slot].item()
        gv = w_gate_f[e * inter : (e + 1) * inter] @ xf[tok]
        uv = w_up_f[e * inter : (e + 1) * inter] @ xf[tok]
        ref[slot] = torch.sigmoid(gv) * gv * uv
    return ref.to(torch.bfloat16)


def _check(
    ref: torch.Tensor,
    test: torch.Tensor,
    label: str,
    atol=0.5,
    rtol=0.05,
    pass_pct=95.0,
):
    delta = (ref.float() - test.float()).abs()
    pct = (
        torch.isclose(ref.float(), test.float(), atol=atol, rtol=rtol)
        .float()
        .mean()
        .item()
        * 100
    )
    assert pct >= pass_pct, (
        f"{label}: only {pct:.1f}% of elements within atol={atol}, rtol={rtol} "
        f"(max_delta={delta.max().item():.4f})"
    )


# ---------------------------------------------------------------------------
# Model shape presets
# ---------------------------------------------------------------------------

SHAPES: List[Tuple[str, int, int, int, int]] = [
    # name,        hidden, inter, topk, experts
    ("qwen3next", 2048, 512, 10, 512),
    ("minimax", 3072, 1536, 8, 256),
    ("deepseek-v3", 7168, 2048, 8, 256),
]

BATCHES = [1, 2, 4]


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_gate_up_bf16_f32path(shape_name, hidden, inter, topk, experts, B):
    """BF16xBF16 gate_up, FP32 scalar path — correct on gfx942 and gfx950."""
    torch.manual_seed(42)
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w_gate = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_up = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")

    ref = _ref_gate_up(x, w_gate, w_up, router_ids, B, topk, inter)

    exe = compile_wd_moe_gate_up(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=False
    )
    _run_kernel(
        exe, inter_out, x, w_gate, w_up, router_ids, B, topk, inter, hidden, experts
    )

    _check(ref, inter_out.view(B * topk, inter), f"{shape_name} B={B} f32path")


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_gate_up_bf16_dot2path(shape_name, hidden, inter, topk, experts, B):
    """BF16xBF16 gate_up, v_dot2_f32_bf16 path — gfx950 only."""
    if not _is_gfx950():
        pytest.skip(f"v_dot2_f32_bf16 requires gfx950, got {_rocm_arch()}")

    torch.manual_seed(42)
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w_gate = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_up = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")

    ref = _ref_gate_up(x, w_gate, w_up, router_ids, B, topk, inter)

    exe = compile_wd_moe_gate_up(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=True
    )
    _run_kernel(
        exe, inter_out, x, w_gate, w_up, router_ids, B, topk, inter, hidden, experts
    )

    _check(ref, inter_out.view(B * topk, inter), f"{shape_name} B={B} dot2path")


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_gate_up_bf16x_fp8w(shape_name, hidden, inter, topk, experts, B):
    """BF16 act x FP8 weight gate_up — gfx950 only, matches CK gate_bf16_d2."""
    if not _is_gfx950():
        pytest.skip(f"w_dtype='fp8' requires gfx950, got {_rocm_arch()}")

    torch.manual_seed(42)
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    # Generate weights as float32 then quantise to OCP FP8 E4M3 for reference.
    wg_f32 = torch.randn(experts * inter, hidden) * 0.1
    wu_f32 = torch.randn(experts * inter, hidden) * 0.1
    # Store as uint8 (raw fp8 bytes) for kernel; dequant to float32 for reference.
    wg_fp8_raw = wg_f32.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    wu_fp8_raw = wu_f32.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
    # Reference uses the dequantised float32 values (round-trip through fp8 format).
    wg_deq = wg_fp8_raw.float().view(torch.float8_e4m3fn).float().cpu()
    wu_deq = wu_fp8_raw.float().view(torch.float8_e4m3fn).float().cpu()
    wg_deq = wg_f32.to(torch.float8_e4m3fn).float()
    wu_deq = wu_f32.to(torch.float8_e4m3fn).float()
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")

    ref = _ref_gate_up_fp8(x, wg_deq.cuda(), wu_deq.cuda(), router_ids, B, topk, inter)

    exe = compile_wd_moe_gate_up(
        hidden=hidden, inter=inter, experts=experts, topk=topk, w_dtype="fp8"
    )
    _run_kernel(
        exe,
        inter_out,
        x,
        wg_fp8_raw,
        wu_fp8_raw,
        router_ids,
        B,
        topk,
        inter,
        hidden,
        experts,
        w_scale=1.0,
    )

    # FP8 quantisation introduces rounding; use relaxed tolerance.
    _check(
        ref,
        inter_out.view(B * topk, inter),
        f"{shape_name} B={B} fp8w",
        atol=0.1,
        rtol=0.1,
        pass_pct=90.0,
    )


# ---------------------------------------------------------------------------
# Benchmark (not collected by pytest by default; run directly)
# ---------------------------------------------------------------------------


def _bench_shape(
    shape_name,
    hidden,
    inter,
    topk,
    experts,
    B,
    warmup=5,
    iters=30,
    w_dtype="bf16",
    use_dot2=False,
):
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    if w_dtype == "fp8":
        wg_raw = torch.randn(experts * inter, hidden) * 0.1
        wu_raw = torch.randn(experts * inter, hidden) * 0.1
        w_gate = wg_raw.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        w_up = wu_raw.to(torch.float8_e4m3fn).view(torch.uint8).cuda()
        w_scale = 1.0
    else:
        w_gate = (
            torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda")
            * 0.1
        )
        w_up = (
            torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda")
            * 0.1
        )
        w_scale = 1.0
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")

    exe = compile_wd_moe_gate_up(
        hidden=hidden,
        inter=inter,
        experts=experts,
        topk=topk,
        w_dtype=w_dtype,
        use_dot2=use_dot2,
    )

    for _ in range(warmup):
        _run_kernel(
            exe,
            inter_out,
            x,
            w_gate,
            w_up,
            router_ids,
            B,
            topk,
            inter,
            hidden,
            experts,
            w_scale,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run_kernel(
            exe,
            inter_out,
            x,
            w_gate,
            w_up,
            router_ids,
            B,
            topk,
            inter,
            hidden,
            experts,
            w_scale,
        )
    ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Arithmetic intensity metrics (weight-bandwidth-bound at decode)
    # x is read once per neuron per slot; w_gate + w_up each once per neuron per slot.
    x_bytes = B * topk * inter * hidden * 2  # bf16
    w_bytes = B * topk * inter * hidden * 2 * 2  # gate + up, bf16
    out_bytes = B * topk * inter * 2  # bf16
    total_bytes = x_bytes + w_bytes + out_bytes
    flops = B * topk * inter * (4 * hidden + 5)

    tag = "dot2" if use_dot2 else "f32 "
    print(
        f"  {shape_name:<14} B={B}  [{tag}]  {ms:7.4f} ms  "
        f"{flops / (ms * 1e9):6.2f} TFLOP/s  {total_bytes / (ms * 1e6):7.1f} GB/s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Warp-decode gate_up benchmark")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--dot2", action="store_true", help="Use v_dot2 path (gfx950 only)"
    )
    args = parser.parse_args()

    arch = _rocm_arch()
    print(f"GPU arch: {arch}")
    if args.dot2 and not _is_gfx950():
        print(f"WARNING: --dot2 requires gfx950, got {arch}. Falling back to f32 path.")
        args.dot2 = False

    print(
        f"\n{'shape':<14} {'B':>3}  {'path':>6}  {'ms':>9}  {'TFLOP/s':>9}  {'GB/s':>9}"
    )
    print("-" * 68)

    for shape_name, hidden, inter, topk, experts in SHAPES:
        for B in args.batches:
            _bench_shape(
                shape_name,
                hidden,
                inter,
                topk,
                experts,
                B,
                warmup=args.warmup,
                iters=args.iters,
                use_dot2=args.dot2,
            )
