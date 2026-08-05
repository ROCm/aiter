#!/usr/bin/env python3
"""Benchmark + correctness harness for the FlyDSL warp-decode gate_up kernel.

Standalone: only requires FlyDSL (no aiter).  Routing is faked with torch.randint
(no moe_sorting, no quantization) so it runs on gfx942 with the bf16xbf16 variant.

Usage:
  python bench_flydsl_wd.py                               # all shapes, B=1..8
  python bench_flydsl_wd.py --shapes deepseek-v3 --batches 1 2 4
  python bench_flydsl_wd.py --shapes qwen3next --batches 1 --verify
  python bench_flydsl_wd.py --iters 50 --warmup 10

Output columns (same as CK bench):
  shape | B | kernel | ms | TFLOP/s | GB/s

--verify runs a torch.matmul reference and prints max_delta / %close.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import torch

# Ensure the harness kernels dir is on the path
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HARNESS_DIR)

import flydsl.compiler as flyc
import flydsl.expr as fx

# Use the aiter kernel (supports both bf16 and fp8 weights).
# Falls back to the local harness stub if aiter is not importable.
try:
    import importlib.util, pathlib as _pl
    _spec = importlib.util.spec_from_file_location(
        "moe_warp_decode",
        _pl.Path(__file__).parents[4] / "aiter/aiter/ops/flydsl/kernels/moe_warp_decode.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    compile_wd_moe_gate_up = _mod.compile_wd_moe_gate_up
    compile_wd_moe_gate_up_splitk = _mod.compile_wd_moe_gate_up_splitk
    compile_wd_moe_gate_finalize = _mod.compile_wd_moe_gate_finalize
    compile_wd_moe_down_reduce = _mod.compile_wd_moe_down_reduce
except Exception:
    from kernels.wd_gate_up_bf16 import compile_wd_moe_gate_up
    compile_wd_moe_gate_up_splitk = None
    compile_wd_moe_gate_finalize = None
    compile_wd_moe_down_reduce = None


_DUMMY_SCALE = None


def _dummy_scale_ptr():
    global _DUMMY_SCALE
    if _DUMMY_SCALE is None:
        _DUMMY_SCALE = torch.zeros(1, dtype=torch.uint8, device="cuda")
    return _ptr(_DUMMY_SCALE)


def _ptr(t: torch.Tensor):
    """Convert a GPU tensor to a raw FlyDSL pointer (fx.Uint8)."""
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

# ── Model shape presets (matches CK bench_warp_decode.cpp) ──────────────────

SHAPES: Dict[str, Dict] = {
    "deepseek-v3": dict(hidden=7168, inter=2048, topk=8, experts=256),
    "minimax":     dict(hidden=3072, inter=1536, topk=8, experts=256),
    "qwen3next":   dict(hidden=2048, inter=512,  topk=10, experts=512),
}


# ── GPU arch detection ────────────────────────────────────────────────────────

def _get_rocm_arch() -> str:
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx" in line:
                return line.split()[-1].strip()
    except Exception:
        pass
    return "unknown"


# ── Correctness reference ─────────────────────────────────────────────────────

def _torch_gate_up_ref(
    x: torch.Tensor,          # [B, hidden] bf16
    w_gate: torch.Tensor,     # [E*INTER, hidden] bf16
    w_up: torch.Tensor,       # [E*INTER, hidden] bf16
    router_ids: torch.Tensor, # [B*TOPK] i32
    B: int,
    topk: int,
    inter: int,
    hidden: int,
) -> torch.Tensor:
    """CPU/GPU torch reference for gate_up (bf16xbf16).

    Returns inter_out: [B*TOPK, inter] bf16.
    """
    out = torch.zeros(B * topk, inter, dtype=torch.float32, device=x.device)
    x_f = x.float()
    w_gate_f = w_gate.float()
    w_up_f = w_up.float()
    for slot in range(B * topk):
        tok = slot // topk
        expert_k = slot % topk
        e = router_ids[slot].item()
        w_row_base = e * inter
        # gate and up: [inter, hidden] slice of weight matrix
        gate_w = w_gate_f[w_row_base : w_row_base + inter]  # [inter, hidden]
        up_w = w_up_f[w_row_base : w_row_base + inter]
        x_tok = x_f[tok]                                    # [hidden]
        gate_v = gate_w @ x_tok                             # [inter]
        up_v = up_w @ x_tok
        # silu(gate) * up
        out[slot] = torch.sigmoid(gate_v) * gate_v * up_v
    return out.to(torch.bfloat16)


def _check_result(ref, test, label, atol=0.5, rtol=0.05, pass_pct=90.0):
    ref_f = ref.float().cpu()
    test_f = test.float().cpu()
    max_delta = (ref_f - test_f).abs().max().item()
    close = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol)
    pct = close.float().mean().item() * 100
    passed = pct >= pass_pct
    status = "PASS" if passed else "FAIL"
    print(
        f"  [{status}] {label}: max_delta={max_delta:.4f}, {pct:.1f}% close "
        f"(atol={atol}, rtol={rtol})"
    )
    if not passed:
        print(f"    ref  sample: {ref_f.reshape(-1)[:8]}")
        print(f"    test sample: {test_f.reshape(-1)[:8]}")
    return passed


# ── Timer ────────────────────────────────────────────────────────────────────

def _gpu_time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


# ── Performance metrics ───────────────────────────────────────────────────────

def _gate_up_flops(B, hidden, inter, topk):
    return B * topk * inter * (4 * hidden + 5)


def _gate_up_bytes_bf16(B, hidden, inter, topk):
    x_bytes = B * topk * inter * hidden * 2
    w_bytes = B * topk * inter * hidden * 2 * 2
    router_bytes = B * topk * 4
    inter_bytes = B * topk * inter * 2
    return x_bytes + w_bytes + router_bytes + inter_bytes


def _down_flops(B, hidden, inter, topk):
    return B * topk * hidden * inter * 2  # MACs: 2 per element


def _down_bytes_bf16(B, hidden, inter, topk):
    inter_bytes = B * topk * inter * 2    # bf16 activations
    w_bytes = B * topk * hidden * inter * 2  # bf16 weights
    y_bytes = B * hidden * 4              # f32 output
    return inter_bytes + w_bytes + y_bytes


def _down_bytes_fp8(B, hidden, inter, topk):
    inter_bytes = B * topk * inter * 2    # bf16 activations
    w_bytes = B * topk * hidden * inter * 1  # fp8 weights
    y_bytes = B * hidden * 4              # f32 output
    return inter_bytes + w_bytes + y_bytes


# ── Single shape benchmark ────────────────────────────────────────────────────

def bench_shape(
    shape_name: str,
    shape: Dict,
    batches: List[int],
    warmup: int,
    iters: int,
    verify: bool,
    arch: str,
):
    hidden = shape["hidden"]
    inter = shape["inter"]
    topk = shape["topk"]
    experts = shape["experts"]

    # use_dot2=True requires gfx950; fall back to FP32 scalar on gfx942
    _use_dot2 = "gfx950" in arch or "gfx95" in arch

    # Compile BF16×BF16 and (on gfx950) BF16×FP8 kernels once.
    torch.manual_seed(42)
    kw = dict(hidden=hidden, inter=inter, experts=experts, topk=topk)
    exe_bf16 = compile_wd_moe_gate_up(**kw, w_dtype="bf16", use_dot2=_use_dot2)
    exe_fp8 = compile_wd_moe_gate_up(**kw, w_dtype="fp8") if _use_dot2 else None

    # BF16 weight tensors (shared across batch sizes)
    w_gate_bf16 = torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    w_up_bf16 = torch.randn(experts * inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.1

    # FP8 weight tensors (quantise from the same floats for a fair byte-count comparison)
    if _use_dot2:
        w_gate_fp8 = w_gate_bf16.float().to(torch.float8_e4m3fn).view(torch.uint8)
        w_up_fp8 = w_up_bf16.float().to(torch.float8_e4m3fn).view(torch.uint8)

    for B in batches:
        x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
        router_ids = torch.randint(0, experts, (B * topk,), dtype=torch.int32, device="cuda")
        inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.current_stream()

        def run_bf16():
            exe_bf16(
                _ptr(inter_out), _ptr(x), _ptr(w_gate_bf16), _ptr(w_up_bf16),
                _ptr(router_ids), B, topk, inter, hidden, experts, 1.0, stream,
            )

        # Correctness check (BF16×BF16 only — reference is exact)
        if verify:
            run_bf16()
            torch.cuda.synchronize()
            ref = _torch_gate_up_ref(x, w_gate_bf16, w_up_bf16, router_ids, B, topk, inter, hidden)
            _check_result(ref, inter_out.view(B * topk, inter), f"{shape_name} B={B}")

        # BF16×BF16 timing
        ms = _gpu_time_ms(run_bf16, warmup=warmup, iters=iters)
        flops = _gate_up_flops(B, hidden, inter, topk)
        gbs = _gate_up_bytes_bf16(B, hidden, inter, topk) / (ms * 1e6)
        print(
            f"  {shape_name:<18} {B:>4}  gate_up_bf16x2        "
            f"{ms:>9.4f}  {flops / (ms * 1e9):>10.2f}  {gbs:>10.1f}"
        )

        # BF16×FP8 timing (gfx950 only)
        if exe_fp8 is not None:
            def run_fp8():
                exe_fp8(
                    _ptr(inter_out), _ptr(x), _ptr(w_gate_fp8), _ptr(w_up_fp8),
                    _ptr(router_ids), B, topk, inter, hidden, experts, 1.0, stream,
                )

            ms_fp8 = _gpu_time_ms(run_fp8, warmup=warmup, iters=iters)
            x_bytes = B * topk * inter * hidden * 2
            w_bytes = B * topk * inter * hidden * 1 * 2   # gate + up fp8
            out_bytes = B * topk * inter * 2
            gbs_fp8 = (x_bytes + w_bytes + out_bytes) / (ms_fp8 * 1e6)
            print(
                f"  {shape_name:<18} {B:>4}  gate_bf16x_fp8w       "
                f"{ms_fp8:>9.4f}  {flops / (ms_fp8 * 1e9):>10.2f}  {gbs_fp8:>10.1f}"
            )

        # ── down_reduce timing ────────────────────────────────────────────────
        if compile_wd_moe_down_reduce is None:
            continue

        # BF16 activations (use inter_out from gate_up as inter_states proxy)
        inter_states = torch.randn(B * topk, inter, dtype=torch.bfloat16, device="cuda") * 0.1
        w_down_bf16 = torch.randn(experts * hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.1
        router_wts = torch.ones(B * topk, dtype=torch.float32, device="cuda") / topk
        y_out = torch.zeros(B, hidden, dtype=torch.float32, device="cuda")

        exe_dn_h2_bf16 = compile_wd_moe_down_reduce(
            **dict(hidden=hidden, inter=inter, experts=experts, topk=topk),
            use_dot2=_use_dot2, h_per_warp=2,
        ) if _use_dot2 else compile_wd_moe_down_reduce(
            **dict(hidden=hidden, inter=inter, experts=experts, topk=topk),
            use_dot2=False, h_per_warp=2,
        )

        def run_dn_bf16():
            y_out.zero_()
            exe_dn_h2_bf16(
                _ptr(y_out), _ptr(inter_states), _ptr(w_down_bf16),
                _dummy_scale_ptr(), _ptr(router_ids), _ptr(router_wts),
                B, topk, inter, hidden, experts, 1.0, stream,
            )

        ms_dn = _gpu_time_ms(run_dn_bf16, warmup=warmup, iters=iters)
        dn_flops = _down_flops(B, hidden, inter, topk)
        gbs_dn = _down_bytes_bf16(B, hidden, inter, topk) / (ms_dn * 1e6)
        tag_dn = "down_h2_d2" if _use_dot2 else "down_h2_f32"
        print(
            f"  {shape_name:<18} {B:>4}  {tag_dn:<22}"
            f"{ms_dn:>9.4f}  {dn_flops / (ms_dn * 1e9):>10.2f}  {gbs_dn:>10.1f}"
        )

        # FP8 down (gfx950 only)
        if _use_dot2:
            w_down_fp8 = w_down_bf16.float().to(torch.float8_e4m3fn).view(torch.uint8).cuda()
            exe_dn_h2_fp8 = compile_wd_moe_down_reduce(
                **dict(hidden=hidden, inter=inter, experts=experts, topk=topk),
                use_dot2=True, h_per_warp=2, w_dtype="fp8",
            )

            def run_dn_fp8():
                y_out.zero_()
                exe_dn_h2_fp8(
                    _ptr(y_out), _ptr(inter_states), _ptr(w_down_fp8),
                    _dummy_scale_ptr(), _ptr(router_ids), _ptr(router_wts),
                    B, topk, inter, hidden, experts, 1.0, stream,
                )

            ms_dn_fp8 = _gpu_time_ms(run_dn_fp8, warmup=warmup, iters=iters)
            gbs_dn_fp8 = _down_bytes_fp8(B, hidden, inter, topk) / (ms_dn_fp8 * 1e6)
            print(
                f"  {shape_name:<18} {B:>4}  down_h2_fp8w_d2       "
                f"{ms_dn_fp8:>9.4f}  {dn_flops / (ms_dn_fp8 * 1e9):>10.2f}  {gbs_dn_fp8:>10.1f}"
            )

        # ── n_waves down_reduce sweep (arch-agnostic, f32 path) ──────────────
        # Sweep n_waves=2,4 to find the best LDS cooperative-load config.
        # Skip n_waves values where inter is not divisible by n_waves*WAVE_SIZE*2.
        if compile_wd_moe_down_reduce is not None:
            for nw in [2, 4]:
                if inter % (nw * 128) != 0 or hidden % (nw * 2) != 0:
                    continue
                exe_dn_nw = compile_wd_moe_down_reduce(
                    **dict(hidden=hidden, inter=inter, experts=experts, topk=topk),
                    use_dot2=False, h_per_warp=2, n_waves=nw,
                )

                def run_dn_nw(exe=exe_dn_nw):
                    y_out.zero_()
                    exe(
                        _ptr(y_out), _ptr(inter_states), _ptr(w_down_bf16),
                        _dummy_scale_ptr(), _ptr(router_ids), _ptr(router_wts),
                        B, topk, inter, hidden, experts, 1.0, stream,
                    )

                ms_nw = _gpu_time_ms(run_dn_nw, warmup=warmup, iters=iters)
                gbs_nw = _down_bytes_bf16(B, hidden, inter, topk) / (ms_nw * 1e6)
                print(
                    f"  {shape_name:<18} {B:>4}  down_h2_f32_nw{nw:<2}      "
                    f"{ms_nw:>9.4f}  {dn_flops / (ms_nw * 1e9):>10.2f}  {gbs_nw:>10.1f}"
                )

        # ── split-K gate_up (two-phase, arch-agnostic) ───────────────────────
        # Benchmark compile_wd_moe_gate_up_splitk for k_batch=2 and k_batch=4.
        # Uses FP32 scratch buffers for gate/up partials.
        if compile_wd_moe_gate_up_splitk is not None:
            for kb in [2, 4]:
                if hidden % (kb * 64 * 8) != 0:
                    continue
                exe_sk = compile_wd_moe_gate_up_splitk(
                    hidden=hidden, inter=inter, experts=experts, topk=topk, k_batch=kb,
                )
                exe_fin = compile_wd_moe_gate_finalize(inter=inter, topk=topk)
                gate_scratch = torch.zeros(B * topk * inter, dtype=torch.float32, device="cuda")
                up_scratch = torch.zeros(B * topk * inter, dtype=torch.float32, device="cuda")

                def run_sk(exe_s=exe_sk, exe_f=exe_fin, gs=gate_scratch, us=up_scratch):
                    gs.zero_()
                    us.zero_()
                    exe_s(
                        _ptr(gs), _ptr(us), _ptr(x), _ptr(w_gate_bf16), _ptr(w_up_bf16),
                        _ptr(router_ids), B, topk, inter, hidden, experts, stream,
                    )
                    exe_f(
                        _ptr(inter_out), _ptr(gs), _ptr(us), B, topk, inter, stream,
                    )

                ms_sk = _gpu_time_ms(run_sk, warmup=warmup, iters=iters)
                flops = _gate_up_flops(B, hidden, inter, topk)
                gbs_sk = _gate_up_bytes_bf16(B, hidden, inter, topk) / (ms_sk * 1e6)
                print(
                    f"  {shape_name:<18} {B:>4}  gate_up_sk_kb{kb:<2}       "
                    f"{ms_sk:>9.4f}  {flops / (ms_sk * 1e9):>10.2f}  {gbs_sk:>10.1f}"
                )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL warp-decode gate_up benchmark"
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=list(SHAPES.keys()),
        choices=list(SHAPES.keys()),
        metavar="SHAPE",
        help=f"Shapes to benchmark. Choices: {list(SHAPES.keys())}",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        metavar="B",
    )
    parser.add_argument("--iters",  type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run torch reference and check correctness before timing",
    )
    args = parser.parse_args()

    arch = _get_rocm_arch()
    print(f"GPU arch: {arch}")

    if "gfx942" in arch:
        print("Note: gfx942 detected — bf16xbf16 path only (fp8 is gfx950-only).")

    print(
        f"\n{'shape':<18} {'B':>4}  {'kernel':<22}"
        f"  {'ms':>9}  {'TFLOP/s':>10}  {'GB/s':>10}"
    )
    print("-" * 80)

    for shape_name in args.shapes:
        shape = SHAPES[shape_name]
        hidden = shape["hidden"]
        inter  = shape["inter"]
        # Skip shapes whose HIDDEN is not divisible by WAVE_SIZE*kVector=512
        if hidden % (64 * 8) != 0:
            print(f"  {shape_name}: skipped (hidden={hidden} not divisible by 512)")
            continue
        bench_shape(
            shape_name=shape_name,
            shape=shape,
            batches=args.batches,
            warmup=args.warmup,
            iters=args.iters,
            verify=args.verify,
            arch=arch,
        )


if __name__ == "__main__":
    main()
