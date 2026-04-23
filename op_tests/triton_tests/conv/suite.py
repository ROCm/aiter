# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from aiter.ops.triton.conv._utils import (
    dynamic_conv_tolerances,
    _out_hw,
    _is_1x1_conv,
    _is_3x3_conv,
    _winograd_tolerances,
    apply_activation,
)
from aiter.ops.triton.conv._launch import _select_3x3_method
from aiter.ops.triton.conv.conv2d import conv2d_nhwc
from ._registry import METHOD_REGISTRY, ORDERED_METHODS
from .bench import run_bench_case


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_error: float
    rel_error: float
    message: str = ""


class TestSuite:
    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        verbose=True,
        bench_enabled=False,
        print_shapes=True,
        layout_mode: str = "both",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self.bench_enabled = bench_enabled
        self.print_shapes = print_shapes
        self.layout_mode = layout_mode
        self.results: List[TestResult] = []
        self.bench_records: List[Dict[str, float]] = []
        self.compare3x3_records: List[Dict] = []
        self.total_flops_tri = 0.0
        self.total_time_tri = 0.0
        self.total_flops_th = 0.0
        self.total_time_th = 0.0
        self.total_flops_tri_e2e = 0.0
        self.total_time_tri_e2e = 0.0

    def check_close(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        K_red: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> TestResult:
        got32 = got.float()
        ref32 = ref.float()
        diff = (got32 - ref32).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rel = max_abs / (float(ref32.abs().max().item()) + 1e-6)
        if rtol is None or atol is None:
            K_est = int(K_red) if K_red is not None else 1024
            rtol_calc, atol_calc = dynamic_conv_tolerances(self.dtype, K_est, ref32)
            rtol = rtol if rtol is not None else rtol_calc
            atol = atol if atol is not None else atol_calc
        try:
            torch.testing.assert_close(got32, ref32, rtol=rtol, atol=atol)
            passed = True
            msg = "OK"
        except AssertionError as e:
            passed = False
            msg = str(e).split("\n")[0]
        res = TestResult(name, passed, max_abs, rel, msg)
        self.results.append(res)
        if self.verbose:
            mark = "✓" if passed else "✗"
            print(f"  {mark} {name:<40} | max_abs={max_abs:.3e} rel={rel:.3e}")
        return res

    def add_bench(
        self,
        name: str,
        flops: float,
        ms_tri: float,
        ms_th: float,
        ms_tri_e2e: float = None,
        triton_kernel: str = "",
        miopen_solver: str = "",
        x_shape: str = "",
        y_shape: str = "",
    ):
        tf_tri = flops / (ms_tri * 1e-3) / 1e12
        tf_th = flops / (ms_th * 1e-3) / 1e12
        has_repack = ms_tri_e2e is not None
        if not has_repack:
            ms_tri_e2e = ms_tri
        tf_tri_e2e = flops / (ms_tri_e2e * 1e-3) / 1e12
        self.bench_records.append(
            {
                "name": name,
                "ms_tri": ms_tri,
                "tflops_tri": tf_tri,
                "ms_torch": ms_th,
                "tflops_torch": tf_th,
                "ms_tri_e2e": ms_tri_e2e,
                "tflops_tri_e2e": tf_tri_e2e,
                "has_repack": has_repack,
                "triton_kernel": triton_kernel,
                "miopen_solver": miopen_solver,
                "x_shape": x_shape,
                "y_shape": y_shape,
            }
        )
        self.total_flops_tri += flops
        self.total_time_tri += ms_tri * 1e-3
        self.total_flops_th += flops
        self.total_time_th += ms_th * 1e-3
        if has_repack:
            self.total_flops_tri_e2e += flops
            self.total_time_tri_e2e += ms_tri_e2e * 1e-3
        else:
            self.total_flops_tri_e2e += flops
            self.total_time_tri_e2e += ms_tri * 1e-3
        kern_info = ""
        if triton_kernel or miopen_solver:
            kern_info = (
                f"\n             Triton kernel: {triton_kernel}"
                if triton_kernel
                else ""
            )
            kern_info += (
                f"\n             MIOpen solver: {miopen_solver}"
                if miopen_solver
                else ""
            )
        repack_info = ""
        if has_repack and abs(ms_tri_e2e - ms_tri) > 0.001:
            repack_info = f"\n             Triton kernel+repack: {ms_tri_e2e:6.3f} ms | {tf_tri_e2e:6.2f} TF/s"
        print(
            f"      Bench: Triton {ms_tri:6.3f} ms | {tf_tri:6.2f} TF/s | Torch {ms_th:6.3f} ms | {tf_th:6.2f} TF/s{kern_info}{repack_info}"
        )

    def summary(self) -> bool:
        total = len(self.results)
        passed = sum(r.passed for r in self.results)
        print("\n" + "=" * 80)
        print(f"TEST SUMMARY: {passed}/{total} passed")
        print("=" * 80)
        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}")
                    print(f"    max_abs={r.max_abs_error:.3e}, rel={r.rel_error:.3e}")
                    if r.message:
                        for line in r.message.rstrip().splitlines() or [""]:
                            print(f"    {line}")
        if self.bench_enabled and self.bench_records:
            print("\n" + "-" * 80)
            print("BENCHMARK SUMMARY")
            print("-" * 80)
            tri_tf = [r["tflops_tri"] for r in self.bench_records]
            tri_tf_e2e = [r["tflops_tri_e2e"] for r in self.bench_records]
            th_tf = [r["tflops_torch"] for r in self.bench_records]
            tri_ms = [r["ms_tri"] for r in self.bench_records]
            th_ms = [r["ms_torch"] for r in self.bench_records]
            has_any_repack = any(r["has_repack"] for r in self.bench_records)

            def s(v):
                return f"mean={statistics.mean(v):6.2f} | median={statistics.median(v):6.2f}"

            def s_ms(v):
                return f"mean={statistics.mean(v):7.3f} | median={statistics.median(v):7.3f}"

            print(f"Triton TFLOPS (kernel)        : {s(tri_tf)}")
            if has_any_repack:
                print(f"Triton TFLOPS (kernel+repack) : {s(tri_tf_e2e)}")
            print(f"Torch  TFLOPS                 : {s(th_tf)}")
            print(f"Triton ms     (kernel)        : {s_ms(tri_ms)}")
            if has_any_repack:
                tri_ms_e2e = [
                    r["ms_tri_e2e"] for r in self.bench_records if r["has_repack"]
                ]
                print(f"Triton ms     (kernel+repack) : {s_ms(tri_ms_e2e)}")
            print(f"Torch  ms                     : {s_ms(th_ms)}")
            print(f"Triton total  (kernel)        : {sum(tri_ms):9.3f} ms")
            if has_any_repack:
                print(
                    f"Triton total  (kernel+repack) : {sum(r['ms_tri_e2e'] for r in self.bench_records):9.3f} ms"
                )
            print(f"Torch  total                  : {sum(th_ms):9.3f} ms")
            print("\nAGGREGATE CONV TFLOPS")
            eff_tri = self.total_flops_tri / max(self.total_time_tri, 1e-12) / 1e12
            eff_th = self.total_flops_th / max(self.total_time_th, 1e-12) / 1e12
            eff_tri_e2e = (
                self.total_flops_tri_e2e / max(self.total_time_tri_e2e, 1e-12) / 1e12
                if has_any_repack
                else 0
            )
            print(
                f"  Triton (kernel)        : {eff_tri:6.2f} TF/s   (sum FLOPs / sum time)"
            )
            if has_any_repack:
                print(
                    f"  Triton (kernel+repack) : {eff_tri_e2e:6.2f} TF/s   (sum FLOPs / sum time, includes input repack)"
                )
            print(
                f"  Torch                  : {eff_th:6.2f} TF/s   (sum FLOPs / sum time)"
            )

            # Per-layer benchmark table
            self._print_layer_table()
            # MIOpen Solver Summary table
            self._print_miopen_solver_table()
            # Overall Performance table
            self._print_overall_perf_table(
                tri_tf, tri_tf_e2e, th_tf, eff_tri, eff_tri_e2e, eff_th
            )

        if self.compare3x3_records:
            self._print_3x3_table()
        return passed == total

    def _print_layer_table(self):
        """Print per-layer benchmark results as a box-drawing table."""
        if not self.bench_records:
            return
        if not any(r.get("x_shape") for r in self.bench_records):
            return

        import re

        has_any_repack = any(r["has_repack"] for r in self.bench_records)
        rows = []
        for i, r in enumerate(self.bench_records):
            name = r["name"]
            # Extract type tag like [3x3], [1x1], [general] from name
            m = re.search(r"\[([\w./]+)\]", name)
            ktype = m.group(1) if m else ""
            # Extract layer description: strip the tag brackets for display
            layer = re.sub(r"\s*\[.*$", "", name).strip()

            x_sh = r.get("x_shape", "")
            y_sh = r.get("y_shape", "")
            shape_str = f"{x_sh}→{y_sh}" if x_sh and y_sh else ""

            tri_tf = r["tflops_tri"]
            tri_tf_e2e = r["tflops_tri_e2e"]
            th_tf = r["tflops_torch"]
            winner = "Triton" if tri_tf > th_tf else "Torch"

            triton_k = r.get("triton_kernel", "")
            miopen_s = r.get("miopen_solver", "")

            row = [str(i), layer, ktype, shape_str, triton_k, miopen_s, f"{tri_tf:.2f}"]
            if has_any_repack:
                row.append(f"{tri_tf_e2e:.2f}")
            row.extend([f"{th_tf:.2f}", winner])
            rows.append(tuple(row))

        hdrs = [
            "#",
            "Layer",
            "Type",
            "Shape",
            "Triton Kernel",
            "MIOpen Solver",
            "Tri Kernel TF/s",
        ]
        if has_any_repack:
            hdrs.append("Tri Kernel+Repack TF/s")
        hdrs.extend(["Torch TF/s", "Winner"])
        hdrs = tuple(hdrs)
        ncols = len(hdrs)
        widths = [
            max(len(hdrs[j]), max(len(rows[i][j]) for i in range(len(rows))))
            for j in range(ncols)
        ]

        def fmt(vals):
            return (
                "│"
                + "│".join(f" {str(v):<{widths[j]}} " for j, v in enumerate(vals))
                + "│"
            )

        sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER BENCHMARK")
        print("=" * 80)
        print(sep_top)
        print(fmt(hdrs))
        print(sep_mid)
        for i, row in enumerate(rows):
            print(fmt(row))
            if i < len(rows) - 1:
                print(sep_mid)
        print(sep_bot)

    def _print_miopen_solver_table(self):
        """Print a table grouping layers by MIOpen solver."""
        from collections import OrderedDict

        solver_layers: dict = OrderedDict()
        solver_algo: dict = {}
        algo_map = {
            "ConvWinoFuryRxS<2-3>": "Winograd Fury F(2,3)",
            "ConvBinWinogradRxSf3x2": "Winograd F(3x3,2x2) binary",
            "GemmFwd1x1_0_1": "GEMM (no workspace)",
            "GemmFwdRest": "GEMM fallback",
        }
        for i, r in enumerate(self.bench_records):
            solver = r.get("miopen_solver", "") or "unknown"
            if solver not in solver_layers:
                solver_layers[solver] = []
                solver_algo[solver] = algo_map.get(solver, solver)
            solver_layers[solver].append(f"L{i}")

        if not any(s for s in solver_layers if s != "unknown"):
            return

        rows = []
        for solver, layers in solver_layers.items():
            algo = solver_algo[solver]
            layer_str = ", ".join(layers)
            if len(layer_str) > 80:
                layer_str = (
                    ", ".join(layers[:10]) + f" ... ({len(layers)} layers total)"
                )
            rows.append((solver, algo, layer_str))

        hdrs = ("MIOpen Solver", "Algorithm Type", "Used For")
        widths = [max(len(hdrs[j]), max(len(r[j]) for r in rows)) for j in range(3)]

        def fmt(vals):
            return (
                "│"
                + "│".join(f" {str(v):<{widths[j]}} " for j, v in enumerate(vals))
                + "│"
            )

        sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

        print("\n" + "=" * 80)
        print("MIOpen SOLVER SUMMARY")
        print("=" * 80)
        print(sep_top)
        print(fmt(hdrs))
        print(sep_mid)
        for i, row in enumerate(rows):
            print(fmt(row))
            if i < len(rows) - 1:
                print(sep_mid)
        print(sep_bot)

    def _print_overall_perf_table(
        self, tri_tf, tri_tf_e2e, th_tf, eff_tri, eff_tri_e2e, eff_th
    ):
        """Print overall performance comparison table."""
        has_any_repack = any(r["has_repack"] for r in self.bench_records)
        tri_wins_kernel = sum(
            1 for r in self.bench_records if r["tflops_tri"] > r["tflops_torch"]
        )
        tri_wins_e2e = sum(
            1 for r in self.bench_records if r["tflops_tri_e2e"] > r["tflops_torch"]
        )
        n = len(self.bench_records)

        rows = [
            (
                "Mean TFLOPS (kernel)",
                f"{statistics.mean(tri_tf):.2f}",
                f"{statistics.mean(th_tf):.2f}",
            ),
        ]
        if has_any_repack:
            rows.append(
                (
                    "Mean TFLOPS (kernel+repack)",
                    f"{statistics.mean(tri_tf_e2e):.2f}",
                    f"{statistics.mean(th_tf):.2f}",
                )
            )
        rows.append(
            (
                "Median TFLOPS (kernel)",
                f"{statistics.median(tri_tf):.2f}",
                f"{statistics.median(th_tf):.2f}",
            )
        )
        if has_any_repack:
            rows.append(
                (
                    "Median TFLOPS (kernel+repack)",
                    f"{statistics.median(tri_tf_e2e):.2f}",
                    f"{statistics.median(th_tf):.2f}",
                )
            )
        rows.append(("Aggregate TFLOPS (kernel)", f"{eff_tri:.2f}", f"{eff_th:.2f}"))
        if has_any_repack:
            rows.append(
                (
                    "Aggregate TFLOPS (kernel+repack)",
                    f"{eff_tri_e2e:.2f}",
                    f"{eff_th:.2f}",
                )
            )
        rows.append(
            (
                "Total kernel time (ms)",
                f"{self.total_time_tri*1e3:.2f}",
                f"{self.total_time_th*1e3:.2f}",
            )
        )
        if has_any_repack:
            total_e2e_ms = sum(r["ms_tri_e2e"] for r in self.bench_records)
            rows.append(
                (
                    "Total kernel+repack time (ms)",
                    f"{total_e2e_ms:.2f}",
                    f"{self.total_time_th*1e3:.2f}",
                )
            )
        rows.append(
            (
                "Layer wins (kernel)",
                f"{tri_wins_kernel}/{n}",
                f"{n - tri_wins_kernel}/{n}",
            )
        )
        if has_any_repack:
            rows.append(
                (
                    "Layer wins (kernel+repack)",
                    f"{tri_wins_e2e}/{n}",
                    f"{n - tri_wins_e2e}/{n}",
                )
            )

        hdrs = ("Metric", "Triton", "PyTorch (MIOpen)")
        widths = [max(len(hdrs[j]), max(len(r[j]) for r in rows)) for j in range(3)]

        def fmt(vals):
            return (
                "│"
                + "│".join(f" {str(v):<{widths[j]}} " for j, v in enumerate(vals))
                + "│"
            )

        sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE")
        print("=" * 80)
        print(sep_top)
        print(fmt(hdrs))
        print(sep_mid)
        for i, row in enumerate(rows):
            print(fmt(row))
            if i < len(rows) - 1:
                print(sep_mid)
        print(sep_bot)

    def _print_3x3_table(self):
        recs = self.compare3x3_records
        all_methods = []
        for mname, entry in METHOD_REGISTRY.items():
            if mname == "default":
                continue
            if any(mname in r["methods"] for r in recs):
                all_methods.append(mname)

        hdr = ["Layer", "Channels"]
        for m in all_methods:
            hdr.append(f"{METHOD_REGISTRY[m].short_name} (TF/s)")
        hdr.append("Torch (TF/s)")
        hdr.append("Winner")
        hdr.append("vs Torch")

        rows = []
        for r in recs:
            row = [r["layer"], r["channels"]]
            best_tf = 0.0
            best_name = ""
            for m in all_methods:
                tf = r["methods"].get(m, None)
                if tf is not None:
                    row.append(f"{tf:6.2f}")
                    if tf > best_tf:
                        best_tf = tf
                        best_name = METHOD_REGISTRY[m].short_name
                else:
                    row.append("  --  ")
            row.append(f"{r['torch_tf']:6.2f}")
            row.append(best_name)
            if r["torch_tf"] > 0:
                pct = (best_tf / r["torch_tf"] - 1) * 100
                row.append(f"{pct:+.1f}%")
            else:
                row.append("  --  ")
            rows.append(row)

        widths = [
            max(len(str(rows[i][j])) for i in range(len(rows))) for j in range(len(hdr))
        ]
        widths = [max(widths[j], len(hdr[j])) for j in range(len(hdr))]

        def fmt_row(vals):
            cells = []
            for j, v in enumerate(vals):
                cells.append(f" {str(v):>{widths[j]}} ")
            return "│" + "│".join(cells) + "│"

        sep_top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
        sep_mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
        sep_bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

        print("\n" + "=" * 80)
        print("3x3 KERNEL COMPARISON TABLE")
        print("=" * 80)
        print(sep_top)
        print(fmt_row(hdr))
        print(sep_mid)
        for i, row in enumerate(rows):
            print(fmt_row(row))
            if i < len(rows) - 1:
                print(sep_mid)
        print(sep_bot)


def _get_tolerances(
    method_name, entry, suite, y_ref, N, C, H, W, K_out, R, S, stride, dilation
):
    """Return (rtol, atol) for a given method, handling winograd and auto-route detection."""
    if entry.is_winograd:
        return _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
    if method_name == "default" and _is_3x3_conv(R, S):
        routed = _select_3x3_method(N, C, H, W, K_out, stride, dilation)
        if routed and "winograd" in routed:
            return _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
    return dynamic_conv_tolerances(suite.dtype, C * R * S, y_ref)


def run_all_methods(
    suite: TestSuite,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    stride,
    padding,
    dilation,
    name: str,
    method: str = "default",
    activation: str = "none",
):
    """Shared dispatch: run selected method(s), check correctness, optionally bench."""
    N, C, H, W = x.shape
    K_out, _, R, S = w.shape

    y_ref = F.conv2d(
        x,
        w,
        b.to(dtype=suite.dtype) if b is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    y_ref = apply_activation(y_ref, activation)

    if suite.print_shapes:
        P, Q = _out_hw(H, W, R, S, stride, padding, dilation)
        if _is_1x1_conv(R, S, dilation):
            kernel_type = "[1x1]"
        elif _is_3x3_conv(R, S):
            kernel_type = "[3x3]"
        else:
            kernel_type = "[general]"
        print(
            f"    {name} {kernel_type}: X{tuple(x.shape)} W{tuple(w.shape)} -> Y{tuple(y_ref.shape)}"
        )

    if suite.layout_mode in ("nchw", "both"):
        methods_to_run = ORDERED_METHODS if method == "all" else [method]
        for m in methods_to_run:
            entry = METHOD_REGISTRY[m]
            if entry.guard_fn and not entry.guard_fn(R, S, stride, dilation, C):
                continue
            y_tri = entry.kernel_fn(
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation=activation,
                out_dtype=suite.dtype,
            )
            rtol, atol = _get_tolerances(
                m, entry, suite, y_ref, N, C, H, W, K_out, R, S, stride, dilation
            )
            suite.check_close(
                f"{name} {entry.bench_tag or '[NCHW]'}",
                y_tri,
                y_ref,
                rtol=rtol,
                atol=atol,
            )
            if suite.bench_enabled and method != "all":
                run_bench_case(
                    suite,
                    x,
                    w,
                    b,
                    stride,
                    padding,
                    dilation,
                    activation,
                    name,
                    layout="nchw",
                    method=m,
                )
        if suite.bench_enabled and method == "all":
            run_bench_case(
                suite,
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation,
                name,
                layout="nchw",
                method="all",
            )

    if suite.layout_mode in ("nhwc", "both"):
        y_nhwc = conv2d_nhwc(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            activation=activation,
            out_dtype=suite.dtype,
        )
        if _is_3x3_conv(R, S):
            nhwc_method = _select_3x3_method(N, C, H, W, K_out, stride, dilation)
            if nhwc_method in ("winograd_f4x3", "winograd_f4x3_cblocked"):
                _r, _a = _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
                suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, rtol=_r, atol=_a)
            else:
                suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, K_red=C * R * S)
        else:
            suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, K_red=C * R * S)
        if suite.bench_enabled:
            run_bench_case(
                suite,
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation,
                name,
                layout="nhwc",
            )
