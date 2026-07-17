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
    from aiter.ops.flydsl.kernels.moe_warp_decode import (
        compile_wd_moe_gate_up,
        compile_wd_moe_gate_up_splitk,
        compile_wd_moe_gate_finalize,
        compile_wd_moe_down_reduce,
    )
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
    compile_wd_moe_gate_up_splitk = _mod.compile_wd_moe_gate_up_splitk
    compile_wd_moe_gate_finalize = _mod.compile_wd_moe_gate_finalize
    compile_wd_moe_down_reduce = _mod.compile_wd_moe_down_reduce


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
# down_reduce helpers + tests
# ---------------------------------------------------------------------------


def _ref_down_reduce(
    inter_states: torch.Tensor,  # [B*TOPK, INTER] bf16
    w_down: torch.Tensor,  # [E*HIDDEN, INTER] bf16
    router_ids: torch.Tensor,  # [B*TOPK] i32
    router_wts: torch.Tensor,  # [B*TOPK] f32
    B: int,
    topk: int,
    inter: int,
    hidden: int,
) -> torch.Tensor:
    """FP32 reference for down_reduce: Y = sum_k(rw_k * (inter_k @ W_down_ek.T))."""
    ref = torch.zeros(B, hidden, dtype=torch.float32, device=inter_states.device)
    xf = inter_states.float()
    wf = w_down.float()
    for b in range(B):
        for k in range(topk):
            slot = b * topk + k
            e = router_ids[slot].item()
            rw = router_wts[slot].item()
            partial = wf[e * hidden : (e + 1) * hidden] @ xf[slot]  # [hidden]
            ref[b] += rw * partial
    return ref


def _run_down_kernel(
    exe,
    y_out,
    inter_states,
    w_down,
    router_ids,
    router_wts,
    B,
    topk,
    inter,
    hidden,
    experts,
):
    stream = torch.cuda.current_stream()
    exe(
        _ptr(y_out),
        _ptr(inter_states),
        _ptr(w_down),
        _ptr(router_ids),
        _ptr(router_wts),
        B,
        topk,
        inter,
        hidden,
        experts,
        stream,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_down_reduce_bf16_f32path(shape_name, hidden, inter, topk, experts, B):
    """BF16 intermediate × BF16 weight down_reduce, FP32 scalar path (gfx942 + gfx950)."""
    torch.manual_seed(42)
    inter_states = (
        torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    # Normalise per token
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_down_reduce(
        inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
    )
    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")  # f32, zero-init

    exe = compile_wd_moe_down_reduce(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=False
    )
    _run_down_kernel(
        exe,
        y_out,
        inter_states,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    _check(ref, y_out, f"{shape_name} B={B} down f32path", atol=0.01, rtol=0.05)


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_down_reduce_bf16_dot2path(shape_name, hidden, inter, topk, experts, B):
    """BF16 intermediate × BF16 weight down_reduce, v_dot2 path (gfx950 only)."""
    if not _is_gfx950():
        pytest.skip(f"v_dot2_f32_bf16 requires gfx950, got {_rocm_arch()}")

    torch.manual_seed(42)
    inter_states = (
        torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_down_reduce(
        inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
    )
    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")

    exe = compile_wd_moe_down_reduce(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=True
    )
    _run_down_kernel(
        exe,
        y_out,
        inter_states,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    _check(ref, y_out, f"{shape_name} B={B} down dot2path", atol=0.01, rtol=0.05)


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_down_reduce_h2_f32path(shape_name, hidden, inter, topk, experts, B):
    """down_reduce H2 layout (2 outputs/wave) f32 path — gfx942 + gfx950."""
    torch.manual_seed(42)
    inter_states = (
        torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_down_reduce(
        inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
    )
    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")

    exe = compile_wd_moe_down_reduce(
        hidden=hidden,
        inter=inter,
        experts=experts,
        topk=topk,
        use_dot2=False,
        h_per_warp=2,
    )
    _run_down_kernel(
        exe,
        y_out,
        inter_states,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    _check(ref, y_out, f"{shape_name} B={B} down_h2 f32path", atol=0.01, rtol=0.05)


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_down_reduce_h2_dot2path(shape_name, hidden, inter, topk, experts, B):
    """down_reduce H2 layout (2 outputs/wave) dot2 path — gfx950 only."""
    if not _is_gfx950():
        pytest.skip(f"v_dot2_f32_bf16 requires gfx950, got {_rocm_arch()}")

    torch.manual_seed(42)
    inter_states = (
        torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_down_reduce(
        inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
    )
    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")

    exe = compile_wd_moe_down_reduce(
        hidden=hidden,
        inter=inter,
        experts=experts,
        topk=topk,
        use_dot2=True,
        h_per_warp=2,
    )
    _run_down_kernel(
        exe,
        y_out,
        inter_states,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    _check(ref, y_out, f"{shape_name} B={B} down_h2 dot2path", atol=0.01, rtol=0.05)


# ---------------------------------------------------------------------------
# End-to-end integration: gate_up (f32) → inter_out → down_reduce (f32)
# ---------------------------------------------------------------------------


def _ref_moe_e2e(
    x, w_gate, w_up, w_down, router_ids, router_wts, B, topk, inter, hidden
):
    """Full MoE block reference: gate_up then down_reduce, FP32 arithmetic."""
    xf, wgf, wuf, wdf = x.float(), w_gate.float(), w_up.float(), w_down.float()
    y = torch.zeros(B, hidden, dtype=torch.float32, device=x.device)
    for slot in range(B * topk):
        tok = slot // topk
        e = router_ids[slot].item()
        rw = router_wts[slot].item()
        gv = wgf[e * inter : (e + 1) * inter] @ xf[tok]
        uv = wuf[e * inter : (e + 1) * inter] @ xf[tok]
        ir = torch.sigmoid(gv) * gv * uv  # BF16 round-trip matches kernel
        y[tok] += rw * (wdf[e * hidden : (e + 1) * hidden] @ ir)
    return y


def _run_e2e(
    exe_gu,
    exe_dn,
    x,
    w_gate,
    w_up,
    w_down,
    router_ids,
    router_wts,
    B,
    topk,
    inter,
    hidden,
    experts,
):
    """Run gate_up then down_reduce and return (inter_out_bf16, y_out_f32)."""
    stream = torch.cuda.current_stream()
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")
    exe_gu(
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
        1.0,
        stream,
    )
    torch.cuda.synchronize()

    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")  # must zero-init
    exe_dn(
        _ptr(y_out),
        _ptr(inter_out),
        _ptr(w_down),
        _ptr(router_ids),
        _ptr(router_wts),
        B,
        topk,
        inter,
        hidden,
        experts,
        stream,
    )
    torch.cuda.synchronize()
    return inter_out, y_out


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts", SHAPES)
@pytest.mark.parametrize("B", BATCHES)
def test_moe_e2e_f32path(shape_name, hidden, inter, topk, experts, B):
    """Full MoE block (gate_up → inter → down_reduce), FP32 paths — gfx942 + gfx950."""
    torch.manual_seed(42)
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w_gate = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_up = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_moe_e2e(
        x, w_gate, w_up, w_down, router_ids, router_wts, B, topk, inter, hidden
    )

    exe_gu = compile_wd_moe_gate_up(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=False
    )
    exe_dn = compile_wd_moe_down_reduce(
        hidden=hidden, inter=inter, experts=experts, topk=topk, use_dot2=False
    )
    _, y_out = _run_e2e(
        exe_gu,
        exe_dn,
        x,
        w_gate,
        w_up,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    # Tolerance is slightly wider than individual-kernel tests because two
    # BF16 rounding steps (gate_up output, then down_reduce input) accumulate.
    _check(ref, y_out, f"{shape_name} B={B} e2e f32path", atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# split-K down_reduce tests  (k_batch > 1, arch-agnostic)
# ---------------------------------------------------------------------------

# k_batch values per shape: inter must be divisible by k_batch * 64 * 8 = k_batch * 512
# qwen3next inter=512:  k_batch=1 only (512 / 512 = 1 step, can't split further)
# minimax    inter=1536: k_batch∈{1,3} (1536/512=3)
# deepseek   inter=2048: k_batch∈{1,2,4} (2048/512=4)
_SPLITK_PARAMS = [
    ("qwen3next", 2048, 512, 10, 512, 1),
    ("minimax", 3072, 1536, 8, 256, 3),
    ("deepseek-v3", 7168, 2048, 8, 256, 2),
    ("deepseek-v3", 7168, 2048, 8, 256, 4),
]


def _ref_down_reduce_multi_b(
    inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
):
    """FP32 reference for down_reduce with correct per-token accumulation."""
    ref = torch.zeros(B, hidden, dtype=torch.float32, device=inter_states.device)
    xf = inter_states.float()
    wf = w_down.float()
    for slot in range(B * topk):
        b = slot // topk
        e = router_ids[slot].item()
        rw = router_wts[slot].item()
        ref[b] += rw * (wf[e * hidden : (e + 1) * hidden] @ xf[slot])
    return ref


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts,k_batch", _SPLITK_PARAMS)
@pytest.mark.parametrize("B", BATCHES)
def test_down_reduce_splitk_f32path(
    shape_name, hidden, inter, topk, experts, k_batch, B
):
    """down_reduce split-K (k_batch>1), FP32 scalar path — gfx942 + gfx950."""
    torch.manual_seed(42)
    inter_states = (
        torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_down = (
        torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(
        0, experts, (B * topk,), dtype=torch.int32, device="cuda"
    )
    router_wts_raw = torch.rand(B * topk, dtype=torch.float32, device="cuda")
    router_wts = (
        router_wts_raw.view(B, topk)
        / router_wts_raw.view(B, topk).sum(dim=1, keepdim=True)
    ).reshape(-1)

    ref = _ref_down_reduce_multi_b(
        inter_states, w_down, router_ids, router_wts, B, topk, inter, hidden
    )
    y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")

    exe = compile_wd_moe_down_reduce(
        hidden=hidden,
        inter=inter,
        experts=experts,
        topk=topk,
        use_dot2=False,
        k_batch=k_batch,
    )
    _run_down_kernel(
        exe,
        y_out,
        inter_states,
        w_down,
        router_ids,
        router_wts,
        B,
        topk,
        inter,
        hidden,
        experts,
    )

    _check(
        ref,
        y_out,
        f"{shape_name} B={B} down kb={k_batch} f32path",
        atol=0.01,
        rtol=0.05,
    )


# ---------------------------------------------------------------------------
# split-K gate_up tests (two-phase: atomicAdd partials + finalize)
# ---------------------------------------------------------------------------

# k_batch must satisfy: hidden % (k_batch * 64 * 8) == 0 and k_batch >= 2.
# qwen3next  hidden=2048: 2048/512=4 → k_batch=2,4 ok
# minimax    hidden=3072: 3072/512=6 → k_batch=2,3 ok
# deepseek   hidden=7168: 7168/512=14 → k_batch=2,7 ok
_SPLITK_GATE_UP_PARAMS = [
    ("qwen3next", 2048, 512, 10, 512, 2),
    ("minimax", 3072, 1536, 8, 256, 2),
    ("deepseek-v3", 7168, 2048, 8, 256, 2),
]


@pytest.mark.parametrize("shape_name,hidden,inter,topk,experts,k_batch", _SPLITK_GATE_UP_PARAMS)
@pytest.mark.parametrize("B", BATCHES)
def test_gate_up_splitk_f32path(shape_name, hidden, inter, topk, experts, k_batch, B):
    """gate_up split-K (two-phase FP32 atomicAdd), arch-agnostic f32 scalar path."""
    torch.manual_seed(42)
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w_gate = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    w_up = (
        torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    router_ids = torch.randint(0, experts, (B * topk,), dtype=torch.int32, device="cuda")

    ref = _ref_gate_up(x, w_gate, w_up, router_ids, B, topk, inter)

    # Phase 1: accumulate FP32 gate/up partials via atomicAdd
    gate_scratch = torch.zeros(B * topk * inter, dtype=torch.float32, device="cuda")
    up_scratch = torch.zeros(B * topk * inter, dtype=torch.float32, device="cuda")
    inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")

    exe_sk = compile_wd_moe_gate_up_splitk(
        hidden=hidden, inter=inter, experts=experts, topk=topk, k_batch=k_batch
    )
    exe_fin = compile_wd_moe_gate_finalize(inter=inter, topk=topk)

    stream = torch.cuda.current_stream()
    exe_sk(
        _ptr(gate_scratch),
        _ptr(up_scratch),
        _ptr(x),
        _ptr(w_gate),
        _ptr(w_up),
        _ptr(router_ids),
        B,
        topk,
        inter,
        hidden,
        experts,
        stream,
    )
    torch.cuda.synchronize()

    # Phase 2: silu(gate) * up → BF16
    exe_fin(
        _ptr(inter_out),
        _ptr(gate_scratch),
        _ptr(up_scratch),
        B,
        topk,
        inter,
        stream,
    )
    torch.cuda.synchronize()

    _check(
        ref,
        inter_out.view(B * topk, inter),
        f"{shape_name} B={B} gate_up_sk kb={k_batch}",
        atol=0.1,
        rtol=0.05,
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
