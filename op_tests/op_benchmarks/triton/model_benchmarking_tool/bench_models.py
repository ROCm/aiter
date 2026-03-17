from abc import ABC, abstractmethod
from contextlib import redirect_stdout, redirect_stderr
from typing import Callable, TypeAlias, Optional
import io
import logging
import shlex
import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import argparse
from triton.runtime.errors import OutOfResources
import aiter.ops.triton.utils._triton.arch_info as arch_info

from op_tests.op_benchmarks.triton.bench_gemm_a16w16 import (
    main as bench_gemm_a16w16_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_a8w8_per_token_scale import (
    main as bench_gemm_a8w8_per_token_scale_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_a8w8_blockscale import (
    main as bench_gemm_a8w8_blockscale_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_afp4wfp4 import (
    main as bench_gemm_afp4wfp4_main,
)
from op_tests.op_benchmarks.triton.bench_batched_gemm_a8w8 import (
    main as bench_batched_gemm_a8w8_main,
)
from op_tests.op_benchmarks.triton.bench_batched_gemm_afp4wfp4 import (
    main as bench_batched_gemm_afp4wfp4_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w8 import (
    main as bench_moe_gemm_a8w8_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w8_blockscale import (
    main as bench_moe_gemm_a8w8_blockscale_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w4 import (
    main as bench_moe_gemm_a8w4_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a4w4 import (
    main as bench_moe_gemm_a4w4_main,
)
from op_tests.op_benchmarks.triton.bench_rmsnorm import main as bench_rmsnorm_main
from op_tests.op_benchmarks.triton.bench_rope import main as bench_rope_main
from op_tests.op_benchmarks.triton.bench_mha import main as bench_mha_main
from op_tests.op_benchmarks.triton.bench_mla_decode import main as bench_mla_main
from op_tests.op_benchmarks.triton.bench_unified_attention import (
    main as bench_unified_attention_main,
)


def disable_aiter_logs() -> None:
    logging.getLogger("aiter").disabled = True


disable_aiter_logs()

KERNEL_DICT: dict[str, Callable[[list[str]], None]] = {
    "gemm_a16w16": bench_gemm_a16w16_main,
    "gemm_a8w8_per_token_scale": bench_gemm_a8w8_per_token_scale_main,
    "gemm_a8w8_blockscale": bench_gemm_a8w8_blockscale_main,
    "gemm_afp4wfp4": bench_gemm_afp4wfp4_main,
    "batched_gemm_a8w8": bench_batched_gemm_a8w8_main,
    "batched_gemm_afp4wfp4": bench_batched_gemm_afp4wfp4_main,
    "moe_op_gemm_a8w8": bench_moe_gemm_a8w8_main,
    "moe_op_gemm_a8w8_blockscale": bench_moe_gemm_a8w8_blockscale_main,
    "moe_op_gemm_a8w4": bench_moe_gemm_a8w4_main,
    "moe_op_gemm_a4w4": bench_moe_gemm_a4w4_main,
    "rmsnorm": bench_rmsnorm_main,
    "rope": bench_rope_main,
    "mha": bench_mha_main,
    "mla": bench_mla_main,
    "unified_attention": bench_unified_attention_main,
}

# Shape dicts from model_shapes.json (int, str values)
ShapeDict: TypeAlias = dict[str, int | str]
# model -> kernel -> list of shapes
ModelShapesData: TypeAlias = dict[str, dict[str, list[ShapeDict]]]
# One benchmark result row (metric value; "B" can be None for GEMM)
ResultRow: TypeAlias = dict[str, int | float | str | None]

ROPE_METRIC_NOTE = (
    "Note: RoPE reports only total flops, i.e. total floating-point operations, not throughput (TFLOPS). "
    "Time measurement is not available because short-running kernels cannot be measured accurately "
    "through triton.testing.do_bench; use rocprof for accurate runtime."
)

MLA_METRIC_NOTE = "Note: MLA benchmark only reports time (ms)."


class KernelHandler(ABC):
    """Base class for kernel-specific benchmark logic and result building."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._kernel: str | None = None
        self._metric: str | None = None
        self._gemm_layout: str | None = None
        self._mha_layout: str | None = None
        self._shape: ShapeDict | None = None
        self._batch_size: int | None = None
        self._seq_len: int | None = None
        self._M: int | None = None  # batch_size * seq_len
        self._tp: int | None = None

    def set_run(
        self,
        model: str,
        kernel: str,
        metric: str,
        gemm_layout: str,
        mha_layout: str,
        tp: int,
    ) -> None:
        """Set run-level parameters (constant for all shapes for this kernel)."""
        self._model = model
        self._kernel = kernel
        self._metric = metric
        self._gemm_layout = gemm_layout
        self._mha_layout = mha_layout
        self._tp = tp

    def set_iteration(self, shape: ShapeDict, batch_size: int, seq_len: int) -> None:
        """Set iteration-level parameters (current shape, batch_size, seq_len). M = batch_size * seq_len."""
        self._shape = shape
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._M = batch_size * seq_len

    def to_str(self) -> str:
        shape_str = (
            ", ".join(f"{k}={v}" for k, v in sorted(self._shape.items()))
            if self._shape is not None
            else "None"
        )

        return (
            f"model={self._model} | "
            f"kernel={self._kernel} | "
            f"batch_size={self._batch_size} seq_len={self._seq_len} M={self._M} | "
            f"shape={{ {shape_str} }}"
        )

    @abstractmethod
    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        """Return shapes adjusted for tensor parallelism."""
        ...

    @abstractmethod
    def build_args(self) -> str:
        """Return args_str for the bench subprocess."""
        ...

    @abstractmethod
    def parse_stdout(self, stdout: str) -> float | str:
        """Parse benchmark stdout and return the numeric result."""
        ...

    @abstractmethod
    def build_result_row(self, bench_result: float | str) -> ResultRow:
        """Build the single result dict from current run/iteration state and bench_result."""
        ...


class GemmKernelHandler(KernelHandler):
    """Handler for GEMM and batched GEMM benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            if s.get("TP_dim") in ("N", "K", "B"):
                key = s["TP_dim"]
                s[key] = max(s[key] // self._tp, 1)
            result.append(s)
        return result

    def build_args(self) -> str:
        shape = self._shape
        M = self._M
        N = shape["N"]
        K = shape["K"]
        if "B" in shape:
            B = shape["B"]
            return f"--shape {B} {M} {N} {K} --metric {self._metric} --layout {self._gemm_layout}"
        return (
            f"--shape {M} {N} {K} --metric {self._metric} --layout {self._gemm_layout}"
        )

    def parse_stdout(self, stdout: str) -> float:
        lines = [line for line in stdout.splitlines() if line.strip()]
        data_line = lines[-1]
        last_row_values = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", data_line)))
        if len(last_row_values) not in (5, 6):
            raise ValueError(f"Unexpected GEMM bench output format: {last_row_values}")
        return last_row_values[-1]

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": None,
            "seq_len": None,
            "B": shape["B"] if "B" in shape else None,
            "M": self._M,
            "N": shape["N"],
            "K": shape["K"],
            "gemm_layout": self._gemm_layout,
            self._metric: bench_result,
        }


class MoeKernelHandler(KernelHandler):
    """Handler for MoE benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        return [{**s, "Dim2": max(s["Dim2"] // self._tp, 1)} for s in shapes]

    def build_args(self) -> str:
        shape = self._shape
        M = self._M
        E = shape["E"]
        dim1 = shape["Dim1"]
        dim2 = shape["Dim2"]
        topk = shape["TopK"]
        return f"--M {M} --shape {dim1} {dim2} --experts {E} {topk}"

    def parse_stdout(self, stdout: str) -> float:
        lines = [line for line in stdout.splitlines() if line.strip()]
        data_line = lines[-1]
        data: dict[str, float] = {}
        for result in data_line.split("|"):
            key, value = result.split(":")
            data[key.strip()] = float(value.strip())
        metric = self._metric
        if metric == "time":
            bench_result = data["Kernel latency (us)"] * 1e-3  # Convert from us to ms
        elif metric == "throughput":
            bench_result = data["TFLOPS"]
        elif metric == "bandwidth":
            bench_result = data["TBPS"] * 1e3  # Convert from TBps to GBps
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return bench_result

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": None,
            "seq_len": None,
            "M": self._M,
            "experts": shape["E"],
            "moe_dim1": shape["Dim1"],
            "moe_dim2": shape["Dim2"],
            "topk": shape["TopK"],
            self._metric: bench_result,
        }


class RmsnormKernelHandler(KernelHandler):
    """Handler for RMSNorm benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        return shapes

    def build_args(self) -> str:
        shape = self._shape
        M = self._M
        N = shape["N"]
        return f"--shape {M} {N} --metric {self._metric}"

    def parse_stdout(self, stdout: str) -> float:
        lines = [line for line in stdout.splitlines() if line.strip()]
        data_line = lines[-1]
        last_row_values = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", data_line)))
        if not last_row_values:
            raise ValueError(f"Unexpected RMSNorm bench output format: {data_line}")
        return last_row_values[-1]

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": None,
            "seq_len": None,
            "M": self._M,
            "N": shape["N"],
            self._metric: bench_result,
        }


class RopeKernelHandler(KernelHandler):
    """Handler for RoPE benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            s["num_heads"] = max(s["num_heads"] // self._tp, 1)
            s["num_kv_heads"] = max(s["num_kv_heads"] // self._tp, 1)
            result.append(s)
        return result

    def build_args(self) -> str:
        # There is no support for two_inputs + bshd layout, so we pass bs*sq as seq_len
        shape = self._shape
        M = self._M
        num_heads = int(shape["num_heads"])
        num_kv_heads = int(shape["num_kv_heads"])
        head_dim = int(shape["head_dim"])
        two_inputs = str(shape["two_inputs"]).lower()
        positions = str(shape["positions"]).lower()
        rotate_style = str(shape["rotate_style"]).lower()
        Q = num_heads // num_kv_heads
        return (
            f"-B 1 -S {M} -H {num_kv_heads} -Q {Q} -D {head_dim} "
            f"--rotate_style {rotate_style} --two_inputs {two_inputs} --pos {positions} -l thd"
        )

    def parse_stdout(self, stdout: str) -> str:
        bench_result = None
        for line in stdout.splitlines():
            if "Total flops" in line:
                nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                if nums:
                    val = float(nums[0])
                    bench_result = f"{val:.6e}"
                    break
        if bench_result is None:
            raise ValueError(f"Unexpected RoPE bench output format: {stdout[:200]!r}")
        return bench_result

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": None,
            "seq_len": self._M,
            "hq": shape["num_heads"],
            "hkv": shape["num_kv_heads"],
            "dqk": None,
            "dv": None,
            "rotary_dim": shape["head_dim"],
            "rotate_style": shape["rotate_style"],
            "rope_total_flops": bench_result,
        }


class MhaKernelHandler(KernelHandler):
    """Handler for MHA forward benchmarks (bench_mha.py)."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            s["hq"] = max(shape["hq"] // self._tp, 1)
            s["hkv"] = max(shape["hkv"] // self._tp, 1)
            result.append(s)
        return result

    def build_args(self) -> str:
        shape = self._shape
        return (
            f"-mode fwd -causal true --layout {self._mha_layout} --dtype bf16 -b {self._batch_size} "
            f"-hq {shape['hq']} -hk {shape['hkv']} -sq {self._seq_len} -sk {self._seq_len} "
            f"-d {shape['dqk']} -dv {shape['dv']} -metric {self._metric}"
        )

    def parse_stdout(self, stdout: str) -> float:
        lines = [line.split() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(
                f"Unexpected MHA bench output: expected at least 3 lines, got {len(lines)}"
            )
        if lines[0] != ["bench_mha:"]:
            raise ValueError(f"Unexpected MHA bench output: first line {lines[0]!r}")
        data = lines[2]
        if len(data) < 7:
            raise ValueError(f"Unexpected MHA bench data line: {data!r}")
        return float(data[6])

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "hq": shape["hq"],
            "hkv": shape["hkv"],
            "dqk": shape["dqk"],
            "dv": shape["dv"],
            "mha_layout": self._mha_layout,
            self._metric: bench_result,
        }


class MlaKernelHandler(KernelHandler):
    """Handler for MLA decode forward benchmarks (bench_mla_decode.py)."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            s["hq"] = max(shape["hq"] // self._tp, 1)
            s["hkv"] = max(shape["hkv"] // self._tp, 1)
            result.append(s)
        return result

    def build_args(self) -> str:
        return (
            f"--model deepseek-V3 --dtype bf16 --tensor-parallelism {self._tp} "
            f"-b {self._batch_size} --seqlen {self._seq_len} -equal_seqlens -causal"
        )

    def parse_stdout(self, stdout: str) -> float:
        lines = [line.split() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(
                f"Unexpected MLA bench output: expected at least 3 lines, got {len(lines)}"
            )
        if lines[0] != ["bench_mla_decode:"]:
            raise ValueError(f"Unexpected MLA bench output: first line {lines[0]!r}")
        data = lines[2]
        if len(data) < 10:
            raise ValueError(f"Unexpected MLA bench data line: {data!r}")
        return float(data[9])

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "hq": shape["hq"],
            "hkv": shape["hkv"],
            "dqk": shape["dqk"],
            "dv": shape["dv"],
            self._metric: bench_result,
        }


class UnifiedAttnKernelHandler(KernelHandler):
    """Handler for unified attention benchmarks (bench_unified_attention.py)."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            s["hq"] = max(shape["hq"] // self._tp, 1)
            s["hkv"] = max(shape["hkv"] // self._tp, 1)
            result.append(s)
        return result

    def build_args(self) -> str:
        shape = self._shape
        block_size = int(shape.get("block_size", 16))
        window_size = int(shape.get("window_size", 0))
        return (
            f"--bs {self._batch_size} --num_heads_q {shape['hq']} --num_heads_k {shape['hkv']} "
            f"--head_size {shape['head_size']} --seq_q_l {self._seq_len} --seq_kv_l {self._seq_len} "
            f"--block_size {block_size} --window_size {window_size} --dtype bf16 --metric {self._metric}"
        )

    def parse_stdout(self, stdout: str) -> float:
        lines = [line.split() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(
                f"Unexpected unified_attention bench output: expected at least 3 lines, got {len(lines)}"
            )
        if lines[0] != ["bench_unified_attention:"]:
            raise ValueError(
                f"Unexpected unified_attention bench output: first line {lines[0]!r}"
            )
        data = lines[2]
        if len(data) < 10:
            raise ValueError(f"Unexpected unified_attention bench data line: {data!r}")
        return float(data[9])

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "hq": shape["hq"],
            "hkv": shape["hkv"],
            "dqk": shape["head_size"],
            "dv": shape["head_size"],
            self._metric: bench_result,
        }


def _get_handler(kernel: str) -> KernelHandler:
    if "moe" in kernel:
        return MoeKernelHandler()
    if kernel == "rmsnorm":
        return RmsnormKernelHandler()
    if kernel == "rope":
        return RopeKernelHandler()
    if kernel == "mha":
        return MhaKernelHandler()
    if kernel == "mla":
        return MlaKernelHandler()
    if kernel == "unified_attention":
        return UnifiedAttnKernelHandler()
    if "gemm" in kernel and "moe" not in kernel:
        return GemmKernelHandler()
    raise ValueError(f"Kernel {kernel} not supported")


def read_json(json_path: str) -> ModelShapesData:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(script_dir, json_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(
            f"Required file not found: {json_path!r}. "
            f"bench_models.py depends on this file for model shape definitions. "
            f"Expected path: {full_path}"
        )
    with open(full_path, "r") as f:
        data: ModelShapesData = json.load(f)
    return data


def call_function(
    bench_fn: Callable[[list[str]], None], handler: KernelHandler
) -> Optional[str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    raw_result: Optional[str] = None

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            bench_fn(shlex.split(handler.build_args()))
        if stderr.getvalue():
            print(f"Standard error stream isn't empty: [{stderr.getvalue()}]")
        else:
            raw_result = stdout.getvalue()
    except OutOfResources as e:
        # Parse the error message to extract required LDS and hardware limit.
        # Expected format: "out of resource: shared memory, Required: XXXX, Hardware limit: XXXX..."
        match = re.search(r"Required:\s*(\d+),\s*Hardware limit:\s*(\d+)", str(e))
        if match:
            required = int(match.group(1))
            hw_limit = int(match.group(2))
            ratio: float = required / hw_limit
            print(
                "Out of LDS on %s: %d / %d (%.1fx)"
                % (
                    handler.to_str(),
                    required,
                    hw_limit,
                    ratio,
                )
            )
        else:
            print("Out of resources while benchmarking %s. %s" % (handler.to_str(), e))

    except Exception as e:
        print(
            "Unexpected error while benchmarking %s. %s: %s"
            % (
                handler.to_str(),
                type(e).__name__,
                e,
            )
        )

    # Close matplotlib figures to silence errors and avoid memory leaks.
    plt.close("all")
    return raw_result


def print_and_save_results(
    results: list[ResultRow], metric: str, output_file: str
) -> None:
    df = pd.DataFrame(results)

    # Exclude metric columns and rope_total_flops from Int64 conversion
    metric_cols = {"time", "throughput", "bandwidth"}
    cols = df.select_dtypes(include="number").columns.difference(
        metric_cols | {"rope_total_flops"}
    )
    df[cols] = df[cols].astype("Int64")

    unit = {"time": "ms", "throughput": "tflops", "bandwidth": "GBps"}
    for m in metric_cols:
        if m in df.columns:
            df[f"{m}({unit[m]})"] = df.pop(m)
    if "rope_total_flops" in df.columns:
        df["total_flops(tflops)"] = df.pop("rope_total_flops")

    # Print results grouped by model and kernel
    for model, idf in df.groupby("Model"):
        print(f"\n=== Model: {model} ===")
        for kernel, jdf in idf.groupby("Kernel"):
            print(f"\nKernel: {kernel}")
            print(
                jdf.drop(columns=["Model", "Kernel"])
                .dropna(axis=1)
                .to_string(index=False)
            )

    if (df["Kernel"] == "rope").any():
        print(f"\n{ROPE_METRIC_NOTE}")

    if (df["Kernel"] == "mla").any():
        print(f"\n{MLA_METRIC_NOTE}")

    # Save results to CSV file
    output_path = (
        f"{os.path.join(os.path.dirname(os.path.realpath(__file__)), output_file)}.csv"
    )
    print(f"\nSaving results to {output_path}...\n")
    df.to_csv(output_path, index=False)


def run_benchmarks(
    data: ModelShapesData,
    batch_sizes: list[int],
    seq_lens: list[int],
    TP: int,
    gemm_layout: str,
    mha_layout: str,
    metric: str,
) -> list[ResultRow]:
    results: list[ResultRow] = []
    for model, kernels in data.items():
        print(f"Running benchmarks for {model}...")
        for kernel, shapes in kernels.items():

            if (
                any(s in kernel for s in ["fp4", "a4", "w4"])
                and not arch_info.is_fp4_avail()
            ):
                print(f"FP4 is not supported on this device. Skipping {kernel}.")
                continue

            if kernel == "moe_op_gemm_a8w8" and arch_info.get_arch() != "gfx950":
                print(
                    f"Float8 x MX is not supported on this device. Skipping {kernel}."
                )
                continue

            bench_fn = KERNEL_DICT[kernel]
            handler = _get_handler(kernel)
            # MLA only reports time (ms); use "time" metric for it
            run_metric = "time" if kernel == "mla" else metric
            handler.set_run(model, kernel, run_metric, gemm_layout, mha_layout, TP)
            tp_shapes = handler.get_tp_shapes(shapes)
            for shape in tp_shapes:
                for batch_size in batch_sizes:
                    for seq_len in seq_lens:
                        handler.set_iteration(shape, batch_size, seq_len)
                        stdout = call_function(bench_fn, handler)
                        if stdout is not None:
                            bench_result = handler.parse_stdout(stdout)
                        else:
                            bench_result = "N/A"
                        results.append(handler.build_result_row(bench_result))
    return results


def parse_arg_list(
    raw_values: list[str],
    parser: argparse.ArgumentParser,
    value_name: str = "Value",
) -> list[int]:
    """Parse a list of integers or start:stop:step ranges."""
    result = []
    for value in raw_values:
        if ":" in value:
            parts = value.split(":")
            if len(parts) != 3:
                parser.error(
                    f"Invalid range '{value}'. " "Ranges must be start:stop:step."
                )

            try:
                start, stop, step = map(int, parts)
            except ValueError:
                parser.error(f"Invalid integers in range '{value}'.")

            if start <= 0 or step <= 0:
                parser.error(f"Values must be positive in range '{value}'.")
            if start > stop:
                parser.error(f"Start must be <= stop in range '{value}'.")
            result.extend(range(start, stop + 1, step))
        else:
            try:
                val = int(value)
            except ValueError:
                parser.error(f"Invalid integer value '{value}'.")

            if val <= 0:
                parser.error(f"{value_name} must be positive.")
            result.append(val)

    return sorted(set(result))  # Remove duplicates and sort


def parse_args(available_models: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model benchmarking tool",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        nargs="+",
        default=["1"],
        help=(
            "Batch size(s) to sweep. Accepts:\n"
            "  Single value:            --batch_size 1\n"
            "  Multiple values:         --batch_size 1 2 4\n"
            "  Range start:stop:step:   --batch_size 1:8:2\n"
            "  Combinations of values and ranges are also accepted.\n"
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--seq_len",
        type=str,
        nargs="+",
        default=["4096"],
        help=(
            "Sequence length(s) to sweep. Accepts:\n"
            "  Single value:            --seq_len 512\n"
            "  Multiple values:         --seq_len 256 512 1024\n"
            "  Range start:stop:step:   --seq_len 128:1024:128\n"
            "  Combinations of values and ranges are also accepted.\n"
            "For non-attention kernels, M = batch_size x seq_len is passed as M.\n"
            "Default: 4096."
        ),
    )
    parser.add_argument(
        "--TP",
        type=int,
        choices=[1, 2, 4, 8],
        default=8,
        help="Tensor parallel size. Default: 8.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["throughput", "bandwidth", "time"],
        default="throughput",
        help=(
            "Metric to report (throughput=TFLOPS, bandwidth=GB/s, time=ms). Default: throughput. "
            "RoPE reports total flops (total floating-point operations) in a separate column (see note in output)."
            "MLA benchmark only reports time (ms)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model name filter: case-insensitive regex matched against model name (default: all models). "
            "e.g. 'llama3' to include only Llama3 family, "
            "'llama|qwen' to include both Llama and Qwen families, "
            "'^(?!.*deepseek)' to exclude DeepSeek family."
            f"\nAvailable models: {', '.join(available_models)}."
        ),
    )
    parser.add_argument(
        "--gemm_layout",
        type=str,
        choices=["TN", "TT", "NN", "NT"],
        default="TN",
        help="GEMM layout. Default: TN.",
    )
    parser.add_argument(
        "--mha_layout",
        type=str,
        choices=["bshd", "thd"],
        default="thd",
        help="Multi-head attention (MHA) layout (bshd or thd). Default: thd.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="bench_results",
        help="Name for the CSV output file. Default: bench_results.",
    )

    args = parser.parse_args()
    args.batch_size = parse_arg_list(args.batch_size, parser, value_name="Batch size")
    args.seq_len = parse_arg_list(args.seq_len, parser, value_name="Sequence length")

    return args


def main() -> None:
    data = read_json("model_shapes.json")
    available_models = list(data.keys())
    args = parse_args(available_models)

    models = args.model
    batch_sizes = args.batch_size
    seq_lens = args.seq_len
    TP = args.TP
    metric = args.metric
    gemm_layout = args.gemm_layout
    mha_layout = args.mha_layout
    output_file = args.output_file

    if models is not None:
        try:
            pattern: re.Pattern[str] = re.compile(models, re.IGNORECASE)
            data = {m: data[m] for m in available_models if pattern.search(m)}
            if not data:
                print("There are no models after filtering by model name.")
                return
        except re.error:
            print(f"Invalid model filter regex: {models!r} - running all models.")

    results = run_benchmarks(
        data, batch_sizes, seq_lens, TP, gemm_layout, mha_layout, metric
    )

    print_and_save_results(results, metric, output_file)


if __name__ == "__main__":
    main()
