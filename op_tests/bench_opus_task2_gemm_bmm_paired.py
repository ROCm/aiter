#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adjacent ABBA benchmark for the Task2 public GEMM/BMM adapters.

The private family adapter and public router share one native module, exact
kid and physical kernel.  Measuring them adjacently in one process isolates
the Python interface delta and gives each round symmetric A/B scheduling
exposure on a shared GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import pathlib
import statistics
import sys
import time
from collections.abc import Callable

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
from torch import Tensor

from aiter import dtypes
from aiter.ops.opus import opus_bmm, opus_gemm
from aiter.ops.opus import gemm_op_a16w16 as a16
from aiter.ops.opus import gemm_op_a8w8 as a8


A16_KID = 200
A16_SPLIT_K = 2
A16_M = 64
A16_N = 64
A16_K = 2048

A8_M = 256
A8_N = 256
A8_K = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _measure_block(
    call: Callable[[], object],
    iters: int,
    stream: torch.cuda.Stream | None = None,
) -> float:
    stream_context = (
        torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    )
    with stream_context:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0 / iters


def _stats(samples: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


class PairedRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rows: list[dict[str, object]] = []
        self.graph_stream = torch.cuda.Stream()

    def emit_pair(
        self,
        *,
        case: str,
        operation: str,
        family: str,
        output_dtype: str,
        shape: list[int],
        kid: int,
        split_k: int | None,
        layer: str = "eager",
        private_call: Callable[[], object],
        public_call: Callable[[], object],
        check: Callable[[], None],
        stream: torch.cuda.Stream | None = None,
        add_graph_control: bool = True,
    ) -> None:
        stream_context = (
            torch.cuda.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        with stream_context:
            for _ in range(self.args.warmup):
                private_call()
                public_call()
        torch.cuda.synchronize()

        private_samples: list[float] = []
        public_samples: list[float] = []
        private_repeat_drift: list[float] = []
        public_repeat_drift: list[float] = []
        round_deltas: list[float] = []
        sequences: list[list[float]] = []
        for _ in range(self.args.rounds):
            private_a1 = _measure_block(private_call, self.args.iters, stream)
            public_b1 = _measure_block(public_call, self.args.iters, stream)
            public_b2 = _measure_block(public_call, self.args.iters, stream)
            private_a2 = _measure_block(private_call, self.args.iters, stream)
            private_value = (private_a1 + private_a2) / 2.0
            public_value = (public_b1 + public_b2) / 2.0
            private_samples.append(private_value)
            public_samples.append(public_value)
            private_repeat_drift.append(
                abs(private_a2 - private_a1) / private_value * 100.0
            )
            public_repeat_drift.append(
                abs(public_b2 - public_b1) / public_value * 100.0
            )
            round_deltas.append((public_value / private_value - 1.0) * 100.0)
            sequences.append([private_a1, public_b1, public_b2, private_a2])

        check()
        private_stats = _stats(private_samples)
        public_stats = _stats(public_samples)
        delta = (
            float(public_stats["median_us"])
            / float(private_stats["median_us"])
            - 1.0
        ) * 100.0
        row = {
            "pass": self.args.pass_id,
            "case": case,
            "layer": layer,
            "operation": operation,
            "family": family,
            "output_dtype": output_dtype,
            "shape": shape,
            "actual_kid": kid,
            "split_k": split_k,
            "warmup": self.args.warmup,
            "rounds": self.args.rounds,
            "iters": self.args.iters,
            "private": private_stats,
            "public": public_stats,
            "delta_pct": delta,
            "median_round_delta_pct": statistics.median(round_deltas),
            "private_repeat_drift_pct": statistics.median(
                private_repeat_drift
            ),
            "public_repeat_drift_pct": statistics.median(public_repeat_drift),
            "max_symmetric_repeat_drift_pct": max(
                private_repeat_drift + public_repeat_drift
            ),
            "round_delta_samples_pct": round_deltas,
            "abba_sequences_us": sequences,
        }
        self.rows.append(row)
        print("PERF_PAIR " + json.dumps(row, sort_keys=True), flush=True)

        if add_graph_control:
            graph_stream = self.graph_stream
            with torch.cuda.stream(graph_stream):
                private_call()
                public_call()
            graph_stream.synchronize()
            private_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(private_graph, stream=graph_stream):
                private_call()
            public_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(public_graph, stream=graph_stream):
                public_call()
            graph_stream.synchronize()

            def check_graph() -> None:
                with torch.cuda.stream(graph_stream):
                    private_graph.replay()
                    public_graph.replay()
                graph_stream.synchronize()
                check()

            self.emit_pair(
                case=case,
                operation=operation,
                family=family,
                output_dtype=output_dtype,
                shape=shape,
                kid=kid,
                split_k=split_k,
                layer="graph_replay",
                private_call=private_graph.replay,
                public_call=public_graph.replay,
                check=check_graph,
                stream=graph_stream,
                add_graph_control=False,
            )

    def run_a16(self) -> None:
        torch.manual_seed(0x950A1602)
        x_bmm = torch.randn(
            (1, A16_M, A16_K), device="cuda", dtype=torch.bfloat16
        )
        w_bmm = torch.randn(
            (1, A16_N, A16_K), device="cuda", dtype=torch.bfloat16
        )
        golden = torch.bmm(x_bmm.float(), w_bmm.float().transpose(1, 2))
        workspace = torch.empty(
            (A16_SPLIT_K, 1, A16_M, A16_N),
            device="cuda",
            dtype=torch.float32,
        )

        for dtype_name, dtype, rtol, atol in (
            ("bf16", torch.bfloat16, 0.03, 0.5),
            ("fp32", torch.float32, 1e-3, 0.05),
        ):
            y_bmm = torch.empty(
                (1, A16_M, A16_N), device="cuda", dtype=dtype
            )

            def private_bmm() -> Tensor:
                return a16._launch_a16w16_bmm(
                    x_bmm,
                    w_bmm,
                    y_bmm,
                    kid=A16_KID,
                    split_k=A16_SPLIT_K,
                    workspace=workspace,
                )

            def public_bmm() -> Tensor:
                return opus_bmm(
                    x_bmm,
                    w_bmm,
                    y_bmm,
                    kid=A16_KID,
                    split_k=A16_SPLIT_K,
                    workspace=workspace,
                )

            def check_bmm() -> None:
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y_bmm.float(), golden, rtol=rtol, atol=atol
                )

            self.emit_pair(
                case=f"a16_bmm_{dtype_name}",
                operation="opus_bmm",
                family="a16w16",
                output_dtype=dtype_name,
                shape=[1, A16_M, A16_N, A16_K],
                kid=A16_KID,
                split_k=A16_SPLIT_K,
                private_call=private_bmm,
                public_call=public_bmm,
                check=check_bmm,
            )

            x_gemm, w_gemm = x_bmm[0], w_bmm[0]
            y_gemm = torch.empty((A16_M, A16_N), device="cuda", dtype=dtype)

            def private_gemm() -> Tensor:
                return a16._launch_a16w16_gemm(
                    x_gemm,
                    w_gemm,
                    y_gemm,
                    kid=A16_KID,
                    split_k=A16_SPLIT_K,
                    workspace=workspace,
                )

            def public_gemm() -> Tensor:
                return opus_gemm(
                    x_gemm,
                    w_gemm,
                    y_gemm,
                    kid=A16_KID,
                    split_k=A16_SPLIT_K,
                    workspace=workspace,
                )

            def check_gemm() -> None:
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y_gemm.float(), golden[0], rtol=rtol, atol=atol
                )

            self.emit_pair(
                case=f"a16_gemm_{dtype_name}",
                operation="opus_gemm",
                family="a16w16",
                output_dtype=dtype_name,
                shape=[A16_M, A16_N, A16_K],
                kid=A16_KID,
                split_k=A16_SPLIT_K,
                private_call=private_gemm,
                public_call=public_gemm,
                check=check_gemm,
            )

    @staticmethod
    def _a8_inputs():
        x = (
            torch.arange(A8_M * A8_K, device="cuda", dtype=torch.int32)
            .remainder(5)
            .sub(2)
            .reshape(1, A8_M, A8_K)
            .to(dtypes.fp8)
        )
        w = (
            torch.arange(A8_N * A8_K, device="cuda", dtype=torch.int32)
            .remainder(7)
            .sub(3)
            .reshape(1, A8_N, A8_K)
            .to(dtypes.fp8)
        )
        x_scale = torch.ones(
            (1, A8_M, A8_K // 128), device="cuda", dtype=torch.float32
        )
        x_scale[:, 1::2].mul_(0.5)
        x_scale[:, :, 1].mul_(2.0)
        w_scale = torch.ones(
            (1, A8_N // 128, A8_K // 128),
            device="cuda",
            dtype=torch.float32,
        )
        w_scale[:, 1].mul_(2.0)
        w_scale[:, :, 1].mul_(0.25)
        return x, w, x_scale, w_scale

    def run_a8_noscale(self, x: Tensor, w: Tensor) -> None:
        launch_x = x[0]
        launch_w = w[0]
        golden = x[0].float() @ w[0].float().T
        y = torch.empty((A8_M, A8_N), device="cuda", dtype=torch.float32)

        def private_call() -> Tensor:
            return a8._launch_a8w8_gemm(
                launch_x, launch_w, y, kid=2
            )

        def public_call() -> Tensor:
            return opus_gemm(launch_x, launch_w, y, kid=2)

        def check() -> None:
            torch.cuda.synchronize()
            torch.testing.assert_close(y, golden, rtol=0, atol=0)

        self.emit_pair(
            case="a8_noscale_gemm",
            operation="opus_gemm",
            family="a8w8",
            output_dtype="fp32",
            shape=[A8_M, A8_N, A8_K],
            kid=2,
            split_k=None,
            private_call=private_call,
            public_call=public_call,
            check=check,
        )

    def run_a8_blockscale(
        self, x: Tensor, w: Tensor, x_scale: Tensor, w_scale: Tensor
    ) -> None:
        golden = torch.zeros((A8_M, A8_N), device="cuda", dtype=torch.float32)
        for block_k in range(A8_K // 128):
            partial = x[
                0, :, block_k * 128 : (block_k + 1) * 128
            ].float() @ w[0, :, block_k * 128 : (block_k + 1) * 128].float().T
            golden.add_(
                partial
                * x_scale[0, :, block_k].unsqueeze(1)
                * w_scale[0, :, block_k].repeat_interleave(128).unsqueeze(0)
            )

        launch_x = x[0]
        launch_w = w[0]
        launch_x_scale = x_scale[0]
        launch_w_scale = w_scale[0]
        y = torch.empty((A8_M, A8_N), device="cuda", dtype=torch.float32)

        def private_call() -> Tensor:
            return a8._launch_a8w8_blockscale_gemm(
                launch_x,
                launch_w,
                y,
                launch_x_scale,
                launch_w_scale,
                kid=1,
            )

        def public_call() -> Tensor:
            return opus_gemm(
                launch_x,
                launch_w,
                y,
                kid=1,
                x_scale=launch_x_scale,
                w_scale=launch_w_scale,
            )

        def check() -> None:
            torch.cuda.synchronize()
            torch.testing.assert_close(y, golden, rtol=0, atol=0)

        self.emit_pair(
            case="a8_blockscale_gemm",
            operation="opus_gemm",
            family="a8w8_blockscale",
            output_dtype="fp32",
            shape=[A8_M, A8_N, A8_K],
            kid=1,
            split_k=None,
            private_call=private_call,
            public_call=public_call,
            check=check,
        )

    def run(self) -> None:
        props = torch.cuda.get_device_properties(0)
        arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
        if arch != "gfx950":
            raise RuntimeError(f"requires gfx950, got {arch!r}")
        module = importlib.import_module("module_deepgemm_opus")
        module_path = pathlib.Path(module.__file__).resolve()
        print(
            "PERF_START "
            + json.dumps(
                {
                    "pass": self.args.pass_id,
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "device": torch.cuda.get_device_name(0),
                    "arch": str(getattr(props, "gcnArchName", None)),
                    "module": str(module_path),
                    "module_sha256": _sha256(module_path),
                    "warmup": self.args.warmup,
                    "rounds": self.args.rounds,
                    "iters": self.args.iters,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        with torch.inference_mode():
            self.run_a16()
            x, w, x_scale, w_scale = self._a8_inputs()
            self.run_a8_noscale(x, w)
            self.run_a8_blockscale(x, w, x_scale, w_scale)

        summaries = {}
        for layer in ("eager", "graph_replay"):
            rows = [row for row in self.rows if row["layer"] == layer]
            private_sum = sum(
                float(row["private"]["median_us"]) for row in rows
            )
            public_sum = sum(
                float(row["public"]["median_us"]) for row in rows
            )
            summaries[layer] = {
                "cases": len(rows),
                "private_sum_us": private_sum,
                "public_sum_us": public_sum,
                "delta_pct": (public_sum / private_sum - 1.0) * 100.0,
                "median_case_delta_pct": statistics.median(
                    float(row["delta_pct"]) for row in rows
                ),
            }
        summary = {
            "pass": self.args.pass_id,
            "cases": len(self.rows),
            "all_correct": True,
            "summaries": summaries,
        }
        print("PERF_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    if "AITER_JIT_DIR" not in os.environ:
        raise RuntimeError("AITER_JIT_DIR must point at the current module")
    PairedRunner(_parse_args()).run()


if __name__ == "__main__":
    main()
