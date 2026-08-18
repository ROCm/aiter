#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adjacent ABBA for old internal and current Torch-owned A16 workspace.

The process imports the preserved baseline pybind module while loading the
current module only as a private C ABI library.  This allows every old/current
measurement to share one process, stream, tensor set and external GPU-load
window without mixing the incompatible Python extension module names.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import os
import pathlib
import statistics
import sys
import time
from collections.abc import Callable

_ROOT = pathlib.Path(
    os.environ.get(
        "OPUS_BENCH_SOURCE_ROOT", pathlib.Path(__file__).resolve().parents[1]
    )
).resolve()
sys.path.insert(0, str(_ROOT))

import torch
from torch import Tensor

import aiter
from aiter.jit.core import compile_ops
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.utility.dtypes import aiter_tensor_t
from aiter.utility.dtypes import _aiter_dtype_id
from csrc.opus_gemm.opus_gemm_common import kernels_list


WORKSPACE_KIDS = tuple(range(200, 224)) + tuple(range(1200, 1224))
WORKSPACE_DTYPES = {"bf16_t": torch.bfloat16, "fp32_t": torch.float32}


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    develop=True,
)
def _baseline_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    bias: Tensor | None,
    kernelId: int,
    splitK: int,
) -> Tensor: ...


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_workspace_init",
    develop=True,
)
def _baseline_workspace_init() -> None: ...


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-module", type=pathlib.Path, required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class CurrentCabi:
    """Steady-state equivalent of the current descriptor-pool C ABI path."""

    def __init__(self, module_path: pathlib.Path) -> None:
        self.module_path = module_path.resolve()
        self.library = ctypes.CDLL(str(self.module_path), mode=ctypes.RTLD_LOCAL)
        self.launch = self.library.opus_gemm_a16w16_launch_cabi
        tensor_ptr = ctypes.POINTER(aiter_tensor_t)
        self.launch.argtypes = [
            tensor_ptr,
            tensor_ptr,
            tensor_ptr,
            tensor_ptr,
            tensor_ptr,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int
        self.get_error = self.library.aiter_get_last_error
        self.get_error.argtypes = []
        self.get_error.restype = ctypes.c_char_p
        self.clear_error = self.library.aiter_clear_last_error
        self.clear_error.argtypes = []
        self.clear_error.restype = None
        abi_version = self.library.aiter_ctypes_abi_version
        abi_version.argtypes = []
        abi_version.restype = ctypes.c_int
        self.abi_version = int(abi_version())
        if self.abi_version < 2:
            raise RuntimeError(f"unsupported current C ABI {self.abi_version}")

        self.xq = aiter_tensor_t()
        self.wq = aiter_tensor_t()
        self.y = aiter_tensor_t()
        self.workspace = aiter_tensor_t()
        self.null_tensor = tensor_ptr()

    @staticmethod
    def _fill(tensor: Tensor, descriptor: aiter_tensor_t) -> aiter_tensor_t:
        shape = tensor.shape
        strides = tensor.stride()
        ndim = len(shape)
        if ndim > 8:
            raise AssertionError(f"aiter_tensor_t supports at most 8 dims: {ndim}")
        descriptor.ptr = tensor.data_ptr()
        descriptor.numel_ = tensor.numel()
        descriptor.ndim = ndim
        descriptor.shape[:ndim] = shape
        descriptor.strides[:ndim] = strides
        descriptor.dtype_ = _aiter_dtype_id(tensor.dtype)
        index = tensor.device.index
        descriptor.device_id = -1 if index is None else index
        return descriptor

    def __call__(
        self,
        xq: Tensor,
        wq: Tensor,
        y: Tensor,
        workspace: Tensor,
        kid: int,
        split_k: int,
    ) -> Tensor:
        xq_descriptor = self._fill(xq, self.xq)
        wq_descriptor = self._fill(wq, self.wq)
        y_descriptor = self._fill(y, self.y)
        workspace_descriptor = self._fill(workspace, self.workspace)
        stream = ctypes.c_void_p(torch.cuda.current_stream(xq.device).cuda_stream)
        status = self.launch(
            ctypes.byref(xq_descriptor),
            ctypes.byref(wq_descriptor),
            ctypes.byref(y_descriptor),
            self.null_tensor,
            ctypes.byref(workspace_descriptor),
            kid,
            split_k,
            stream,
        )
        if status != 0:
            raw_error = self.get_error()
            message = (
                raw_error.decode(errors="replace")
                if raw_error
                else f"ctypes status={status}"
            )
            self.clear_error()
            raise RuntimeError(message)
        return y


_CURRENT_CABI: CurrentCabi | None = None


def _paired_current_a16w16_cabi(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    if bias is not None or workspace is None:
        raise ValueError("paired Task1 benchmark requires workspace and no bias")
    if _CURRENT_CABI is None:
        raise RuntimeError("current C ABI is not initialized")
    _CURRENT_CABI(XQ, WQ, Y, workspace, kid, split_k)


def _paired_current_a16w16_cabi_fake(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None:
    return None


_paired_current_a16w16_cabi = torch_compile_guard(
    device="cuda",
    calling_func_=_paired_current_a16w16_cabi,
    gen_fake=_paired_current_a16w16_cabi_fake,
)(_paired_current_a16w16_cabi)


def _measure_block(
    call: Callable[[], object], iters: int, stream: torch.cuda.Stream | None
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


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.current: CurrentCabi | None = None
        self.rows: list[dict[str, object]] = []

    def emit_pair(
        self,
        *,
        kid: int,
        dtype_name: str,
        layer: str,
        shape: list[int],
        baseline_call: Callable[[], object],
        current_call: Callable[[], object],
        check: Callable[[], None],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        stream_context = (
            torch.cuda.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        with stream_context:
            for _ in range(self.args.warmup):
                baseline_call()
                current_call()
        torch.cuda.synchronize()

        baseline_samples: list[float] = []
        current_samples: list[float] = []
        round_deltas: list[float] = []
        baseline_drift: list[float] = []
        current_drift: list[float] = []
        sequences: list[list[float]] = []
        for _ in range(self.args.rounds):
            old_a1 = _measure_block(baseline_call, self.args.iters, stream)
            new_b1 = _measure_block(current_call, self.args.iters, stream)
            new_b2 = _measure_block(current_call, self.args.iters, stream)
            old_a2 = _measure_block(baseline_call, self.args.iters, stream)
            old_value = (old_a1 + old_a2) / 2.0
            new_value = (new_b1 + new_b2) / 2.0
            baseline_samples.append(old_value)
            current_samples.append(new_value)
            baseline_drift.append(abs(old_a2 - old_a1) / old_value * 100.0)
            current_drift.append(abs(new_b2 - new_b1) / new_value * 100.0)
            round_deltas.append((new_value / old_value - 1.0) * 100.0)
            sequences.append([old_a1, new_b1, new_b2, old_a2])

        check()
        old_stats = _stats(baseline_samples)
        new_stats = _stats(current_samples)
        row = {
            "pass": self.args.pass_id,
            "kid": kid,
            "dtype": dtype_name,
            "layer": layer,
            "shape": shape,
            "warmup": self.args.warmup,
            "rounds": self.args.rounds,
            "iters": self.args.iters,
            "baseline": old_stats,
            "current": new_stats,
            "delta_pct": (
                float(new_stats["median_us"]) / float(old_stats["median_us"])
                - 1.0
            )
            * 100.0,
            "median_round_delta_pct": statistics.median(round_deltas),
            "baseline_repeat_drift_pct": statistics.median(baseline_drift),
            "current_repeat_drift_pct": statistics.median(current_drift),
            "max_symmetric_repeat_drift_pct": max(
                baseline_drift + current_drift
            ),
            "round_delta_samples_pct": round_deltas,
            "abba_sequences_us": sequences,
        }
        self.rows.append(row)
        print("PERF_PAIR " + json.dumps(row, sort_keys=True), flush=True)

    def run(self) -> None:
        props = torch.cuda.get_device_properties(0)
        arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
        if arch != "gfx950":
            raise RuntimeError(f"requires gfx950, got {arch!r}")

        # Force the preserved extension to load before the current C ABI is used.
        x_prime = torch.zeros((1, 64, 2048), device="cuda", dtype=torch.bfloat16)
        w_prime = torch.zeros((1, 64, 2048), device="cuda", dtype=torch.bfloat16)
        y_prime = torch.empty((1, 64, 64), device="cuda", dtype=torch.bfloat16)
        _baseline_raw(x_prime, w_prime, y_prime, None, 200, 2)
        module = importlib.import_module("module_deepgemm_opus")
        baseline_module_path = pathlib.Path(module.__file__).resolve()
        self.current = CurrentCabi(self.args.current_module)
        global _CURRENT_CABI
        _CURRENT_CABI = self.current

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
                    "baseline_source": str(pathlib.Path(aiter.__file__).resolve()),
                    "baseline_module": str(baseline_module_path),
                    "baseline_module_sha256": _sha256(baseline_module_path),
                    "current_module": str(self.current.module_path),
                    "current_module_sha256": _sha256(self.current.module_path),
                    "current_cabi_version": self.current.abi_version,
                    "kids": list(WORKSPACE_KIDS),
                    "warmup": self.args.warmup,
                    "rounds": self.args.rounds,
                    "iters": self.args.iters,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            _baseline_workspace_init()
        graph_stream.synchronize()

        with torch.inference_mode():
            for kid in WORKSPACE_KIDS:
                instance = kernels_list[kid]
                m, n, k = (
                    int(instance.B_M),
                    int(instance.B_N),
                    32 * int(instance.B_K),
                )
                torch.manual_seed(0x950000 + kid)
                xq = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16)
                wq = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
                golden = torch.bmm(xq.float(), wq.float().transpose(1, 2))
                workspace = torch.empty(
                    (2, 1, m, n),
                    device="cuda",
                    dtype=WORKSPACE_DTYPES[instance.splitk_workspace_dtype],
                )
                output_cases = [("bf16", torch.bfloat16, 0.03, 0.5)]
                if instance.splitk_workspace_dtype != "bf16_t":
                    output_cases.append(("fp32", torch.float32, 1e-3, 0.05))

                for dtype_name, dtype, rtol, atol in output_cases:
                    old_y = torch.empty((1, m, n), device="cuda", dtype=dtype)
                    new_y = torch.empty((1, m, n), device="cuda", dtype=dtype)

                    def baseline_call() -> Tensor:
                        return _baseline_raw(xq, wq, old_y, None, kid, 2)

                    def current_call() -> Tensor:
                        _paired_current_a16w16_cabi(
                            xq, wq, new_y, None, workspace, kid, 2
                        )
                        return new_y

                    def check_eager() -> None:
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            old_y.float(), golden, rtol=rtol, atol=atol
                        )
                        torch.testing.assert_close(
                            new_y.float(), golden, rtol=rtol, atol=atol
                        )

                    common = {
                        "kid": kid,
                        "dtype_name": dtype_name,
                        "shape": [1, m, n, k],
                    }
                    self.emit_pair(
                        layer="eager",
                        baseline_call=baseline_call,
                        current_call=current_call,
                        check=check_eager,
                        **common,
                    )

                    with torch.cuda.stream(graph_stream):
                        baseline_call()
                        current_call()
                    graph_stream.synchronize()
                    old_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(old_graph, stream=graph_stream):
                        baseline_call()
                    new_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(new_graph, stream=graph_stream):
                        current_call()
                    graph_stream.synchronize()

                    def check_graph() -> None:
                        with torch.cuda.stream(graph_stream):
                            old_graph.replay()
                            new_graph.replay()
                        graph_stream.synchronize()
                        torch.testing.assert_close(
                            old_y.float(), golden, rtol=rtol, atol=atol
                        )
                        torch.testing.assert_close(
                            new_y.float(), golden, rtol=rtol, atol=atol
                        )

                    self.emit_pair(
                        layer="graph_replay",
                        baseline_call=old_graph.replay,
                        current_call=new_graph.replay,
                        check=check_graph,
                        stream=graph_stream,
                        **common,
                    )

        summaries = {}
        for layer in ("eager", "graph_replay"):
            rows = [row for row in self.rows if row["layer"] == layer]
            old_sum = sum(float(row["baseline"]["median_us"]) for row in rows)
            new_sum = sum(float(row["current"]["median_us"]) for row in rows)
            summaries[layer] = {
                "cases": len(rows),
                "baseline_sum_us": old_sum,
                "current_sum_us": new_sum,
                "delta_pct": (new_sum / old_sum - 1.0) * 100.0,
                "median_case_delta_pct": statistics.median(
                    float(row["delta_pct"]) for row in rows
                ),
            }
        print(
            "PERF_COMPLETE "
            + json.dumps(
                {
                    "pass": self.args.pass_id,
                    "all_correct": True,
                    "cases": len(self.rows),
                    "summaries": summaries,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    if "AITER_JIT_DIR" not in os.environ:
        raise RuntimeError("AITER_JIT_DIR must point at the baseline module")
    Runner(_parse_args()).run()


if __name__ == "__main__":
    main()
