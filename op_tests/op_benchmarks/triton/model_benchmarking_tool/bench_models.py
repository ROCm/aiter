import argparse
import io
import json
import logging
import math
import os
import re
import shlex
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
from triton.runtime.errors import OutOfResources

from aiter.ops.triton.gated_delta_net import fused_gdn_prefill_qkvz as gdn_prefill
from aiter.ops.triton.utils._triton import arch_info
from op_tests.op_benchmarks.triton.bench_attn_res import main as bench_attn_res_main
from op_tests.op_benchmarks.triton.bench_batched_gemm_a8w8 import (
    main as bench_batched_gemm_a8w8_main,
)
from op_tests.op_benchmarks.triton.bench_batched_gemm_a16wfp4 import (
    main as bench_batched_gemm_a16wfp4_main,
)
from op_tests.op_benchmarks.triton.bench_batched_gemm_afp4wfp4 import (
    main as bench_batched_gemm_afp4wfp4_main,
)
from op_tests.op_benchmarks.triton.bench_chunk_kda import (
    K3_HEAD_DIM as KDA_PREFILL_HEAD_DIM,
)
from op_tests.op_benchmarks.triton.bench_chunk_kda import (
    main as bench_chunk_kda_main,
)
from op_tests.op_benchmarks.triton.bench_fused_gdn_decode_qkvz import (
    main as bench_fused_gdn_decode_main,
)
from op_tests.op_benchmarks.triton.bench_fused_gdn_prefill_qkvz import (
    main as bench_fused_gdn_prefill_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_a8w8_blockscale import (
    main as bench_gemm_a8w8_blockscale_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_a8w8_per_token_scale import (
    main as bench_gemm_a8w8_per_token_scale_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_a16w16 import (
    main as bench_gemm_a16w16_main,
)
from op_tests.op_benchmarks.triton.bench_gemm_afp4wfp4 import (
    main as bench_gemm_afp4wfp4_main,
)
from op_tests.op_benchmarks.triton.bench_kda import main as bench_kda_main
from op_tests.op_benchmarks.triton.bench_mha import main as bench_mha_main
from op_tests.op_benchmarks.triton.bench_mhc_dsv4 import main as bench_mhc_dsv4_main
from op_tests.op_benchmarks.triton.bench_mla_decode import main as bench_mla_main
from op_tests.op_benchmarks.triton.bench_moe_gemm_a4w4 import (
    main as bench_moe_gemm_a4w4_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w4 import (
    main as bench_moe_gemm_a8w4_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w8 import (
    main as bench_moe_gemm_a8w8_main,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_a8w8_blockscale import (
    main as bench_moe_gemm_a8w8_blockscale_main,
)
from op_tests.op_benchmarks.triton.bench_pa_decode_sparse import (
    main as bench_pa_decode_sparse_main,
)
from op_tests.op_benchmarks.triton.bench_rmsnorm import main as bench_rmsnorm_main
from op_tests.op_benchmarks.triton.bench_rope import main as bench_rope_main
from op_tests.op_benchmarks.triton.bench_unified_attention import DEFAULT_BLOCK_SIZE
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
    "batched_gemm_a16wfp4": bench_batched_gemm_a16wfp4_main,
    "moe_op_gemm_a8w8": bench_moe_gemm_a8w8_main,
    "moe_op_gemm_a8w8_blockscale": bench_moe_gemm_a8w8_blockscale_main,
    "moe_op_gemm_a8w4": bench_moe_gemm_a8w4_main,
    "moe_op_gemm_a4w4": bench_moe_gemm_a4w4_main,
    "rmsnorm": bench_rmsnorm_main,
    # Fused RMSNorm + residual add + MXFP4 quant. Reuses bench_rmsnorm.py
    # (via its --quant mxfp4 mode), so there is no separate bench script.
    "fused_rms_mxfp4_quant": bench_rmsnorm_main,
    "rope": bench_rope_main,
    "mha": bench_mha_main,
    "mla": bench_mla_main,
    "mhc": bench_mhc_dsv4_main,
    "kda_attention_prefill": bench_chunk_kda_main,
    "kda_attention_decode": bench_kda_main,
    "attn_res": bench_attn_res_main,
    "gated_delta_net_prefill": bench_fused_gdn_prefill_main,
    "gated_delta_net_decode": bench_fused_gdn_decode_main,
    "sparse_mqa": bench_pa_decode_sparse_main,
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

# Decode kernels that use batch_size as token concurrency and take their
# context or state size from the shape, so only the first seq_len runs.
DECODE_ONLY_KERNELS = ("sparse_mqa", "gated_delta_net_decode", "kda_attention_decode")

# model_shapes.json model name -> bench_mla_decode.py --model value
# (<family>-<size> from utils/model_configs.json). The MLA benchmark is skipped
# for models that aren't in this mapping.
MLA_BENCH_MODELS: dict[str, str] = {
    "DeepSeek-R1": "deepseek-V3",
    "Kimi-K2 Thinking": "kimik25-tp4",
}


def _last_number(stdout: str, label: str) -> float:
    """Return the last number on the last non-empty line of a bench's stdout."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    values = re.findall(r"-?\d+(?:\.\d+)?", lines[-1]) if lines else []
    if not values:
        last_line = lines[-1] if lines else ""
        raise ValueError(f"Unexpected {label} bench output format: {last_line!r}")
    return float(values[-1])


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

    def _shard(self, value: int) -> int:
        return max(value // self._tp, 1)

    def _shard_keys(self, s: dict, keys: list[str]) -> None:
        for key in keys:
            s[key] = self._shard(s[key])

    @abstractmethod
    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        """Return shapes adjusted for tensor parallelism."""
        ...

    def unsupported_reason(self, shape: ShapeDict) -> str | None:
        """Return why the bench can't run this TP-adjusted shape, or None."""
        return None

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
                s[key] = self._shard(s[key])
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
            "B": shape.get("B", None),
            "M": self._M,
            "N": shape["N"],
            "K": shape["K"],
            "gemm_layout": self._gemm_layout,
            self._metric: bench_result,
        }


# MoE benchmarks print one line per batch. Each layer group is
# "<name> <us>us <TF/s> TF/s <TB/s> TB/s", optionally followed by " [kernel]".
# a4w4 omits that bracket and prints the kernel name once after the groups.
# `total` is both GEMMs back to back, the whole MLP.
# The bench prints a fixed number of significant digits, so a large value can end
# in a bare decimal point ("1071.") and the fraction digits are optional.
_MOE_TOTAL_RE = re.compile(
    r"total\s+"
    r"(?P<us>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)us\s+"
    r"(?P<tflops>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s+TF/s\s+"
    r"(?P<tbps>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s+TB/s"
)


class MoeKernelHandler(KernelHandler):
    """Handler for MoE benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        return [{**s, "Dim2": self._shard(s["Dim2"])} for s in shapes]

    def build_args(self) -> str:
        shape = self._shape
        M = self._M
        E = shape["E"]
        dim1 = shape["Dim1"]
        dim2 = shape["Dim2"]
        topk = shape["TopK"]
        return f"--M {M} --shape {dim1} {dim2} --experts {E} {topk}"

    def parse_stdout(self, stdout: str) -> float:
        matches = list(_MOE_TOTAL_RE.finditer(stdout))
        if not matches:
            raise ValueError(f"Unexpected MoE bench output format: {stdout!r}")
        measurement = matches[-1]
        us = float(measurement.group("us"))
        tflops = float(measurement.group("tflops"))
        tbps = float(measurement.group("tbps"))
        metric = self._metric
        if metric == "time":
            return us * 1e-3  # us -> ms
        if metric == "throughput":
            return tflops
        if metric == "bandwidth":
            return tbps * 1e3  # TB/s -> GB/s
        raise ValueError(f"Unknown metric: {metric}")

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
        return _last_number(stdout, "RMSNorm")

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


class FusedRmsMxfp4QuantKernelHandler(RmsnormKernelHandler):
    """Handler for fused RMSNorm + residual add + MXFP4 quant.

    Identical shape handling to RMSNorm (reads N from model_shapes.json); only
    the bench args differ, flipping bench_rmsnorm.py into its fused-quant mode.
    """

    def build_args(self) -> str:
        return super().build_args() + " --quant mxfp4 --add-residual"


class RopeKernelHandler(KernelHandler):
    """Handler for RoPE benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["num_heads", "num_kv_heads"])
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
            self._shard_keys(s, ["hq", "hkv"])
            result.append(s)
        return result

    def build_args(self) -> str:
        shape = self._shape
        # bshd uses fwd. thd uses fwd_varlen with equal sequence lengths.
        fn = "fwd_varlen" if self._mha_layout == "thd" else "fwd"
        window_left = int(shape.get("sliding_window_left", -1))
        window_right = int(shape.get("sliding_window_right", -1))
        sink = shape.get("sink", None)
        # Pass both window bounds. An omitted window option in bench_mha.py runs its default sweep
        # (dense plus a representative causal sliding window case).
        # Leave -backend / -impl at bench_mha defaults (triton / default): native
        # Triton supports a left window and a sink. A right window needs dao_ai,
        # which rejects sinks; no model shape currently uses a right window.
        args = (
            f"-fn {fn} -causal true --dtype bf16 -b {self._batch_size} "
            f"-hq {shape['hq']} -hk {shape['hkv']} -sq {self._seq_len} -sk {self._seq_len} "
            f"-d {shape['dqk']} -dv {shape['dv']} "
            f"--window-size-left {window_left} --window-size-right {window_right} "
            f"--metric {self._metric}"
        )
        if fn == "fwd_varlen":
            args += " -equal_seqlens"
        if sink:
            args += " -sink"
        return args

    def parse_stdout(self, stdout: str) -> float:
        # Expected output (4 lines):
        #   [0] "[1/1] <model> B=... HQ=... ..."   (progress)
        #   [1] "bench_mha:"
        #   [2] "model  BATCH  HQ  HK  N_CTX_Q  N_CTX_K  D_HEAD  D_HEAD_V  ..."   (header)
        #   [3] "0  <model>  <b>  <hq>  <hk>  <sq>  <sk>  ...  <value>"   (data)
        lines = [line.split() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 4:
            raise ValueError(
                f"Unexpected MHA bench output: expected at least 4 lines, got {len(lines)}"
            )
        if lines[1] != ["bench_mha:"]:
            raise ValueError(f"Unexpected MHA bench output: second line {lines[1]!r}")
        data = lines[3]
        if len(data) < 15:
            raise ValueError(f"Unexpected MHA bench data line: {data!r}")
        return float(data[-1])

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
            "sink": bool(shape.get("sink", False)),
            "sliding_window": shape.get("sliding_window_left", None),
            self._metric: bench_result,
        }


class MlaKernelHandler(KernelHandler):
    """Handler for MLA decode forward benchmarks (bench_mla_decode.py)."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["hq", "hkv"])
            result.append(s)
        return result

    def build_args(self) -> str:
        return (
            f"--model {MLA_BENCH_MODELS[self._model]} --dtype bf16 "
            f"--tensor-parallelism {self._tp} "
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
        # Columns: row_idx, model, B, H, S, kv_lora_rank, qk_rope_head_dim,
        #          rotary_dim, num_kv_splits, <time>.
        # The benchmark takes the head count from model_configs.json, so check
        # it against the head count from model_shapes.json.
        if int(float(data[3])) != self._shape["hq"]:
            raise ValueError(
                f"MLA bench head count H={data[3]} doesn't match hq={self._shape['hq']}"
            )
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


class SparseMqaKernelHandler(KernelHandler):
    """Handler for DeepSeek-V4 sparse MQA decode benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["hq"])
            result.append(s)
        return result

    def unsupported_reason(self, shape: ShapeDict) -> str | None:
        hkv = int(shape.get("hkv", 1))
        dqk = int(shape["dqk"])
        dv = int(shape.get("dv", dqk))
        if hkv != 1 or dv != dqk:
            return (
                "the sparse MQA bench runs one shared KV head with one head "
                f"dimension; this shape has hkv={hkv}, dqk={dqk}, dv={dv}"
            )
        return None

    def build_args(self) -> str:
        shape = self._shape
        return (
            f"--shape {self._batch_size} {shape['hq']} {shape['dqk']} {shape['kv_len']} "
            "--providers v4_a8w8 --pool paged "
            f"--page {shape.get('page', 64)} --extra-len {shape['extra_len']} "
            f"--extra-page {shape.get('extra_page', 64)} "
            f"--metric {self._metric} --timer profiler"
        )

    def parse_stdout(self, stdout: str) -> float:
        result = _last_number(stdout, "sparse_mqa")
        if self._metric == "time":
            return result * 1e-3  # us -> ms
        if self._metric == "bandwidth":
            return result * 1e3  # TB/s -> GB/s
        return result

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": int(shape["kv_len"]) + int(shape["extra_len"]),
            "hq": shape["hq"],
            "hkv": shape.get("hkv", 1),
            "dqk": shape["dqk"],
            "dv": shape.get("dv", shape["dqk"]),
            "kv_len": shape["kv_len"],
            "extra_len": shape["extra_len"],
            "page": shape.get("page", 64),
            "extra_page": shape.get("extra_page", 64),
            self._metric: bench_result,
        }


class MhcKernelHandler(KernelHandler):
    """Handler for DeepSeek-V4 manifold-constrained hyper-connections."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        return [shape.copy() for shape in shapes]

    def build_args(self) -> str:
        shape = self._shape
        return (
            f"-M {self._M} -n {shape['n']} -C {shape['C']} "
            f"--op {shape['op']} --dtype bf16 --metric {self._metric}"
        )

    def parse_stdout(self, stdout: str) -> float:
        return _last_number(stdout, "mhc")

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "M": self._M,
            "n": shape["n"],
            "C": shape["C"],
            "mhc_op": shape["op"],
            self._metric: bench_result,
        }


class KdaAttentionKernelHandler(KernelHandler):
    """Handler for Kimi Delta Attention prefill and decode benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["hq"])
            result.append(s)
        return result

    def unsupported_reason(self, shape: ShapeDict) -> str | None:
        dqk = int(shape["dqk"])
        dv = int(shape["dv"])
        if self._kernel == "kda_attention_prefill":
            if dqk == dv == KDA_PREFILL_HEAD_DIM:
                return None
            return (
                "the KDA prefill bench supports only "
                f"dqk=dv={KDA_PREFILL_HEAD_DIM}; this shape has dqk={dqk}, dv={dv}"
            )
        if dv != dqk:
            return (
                "the KDA decode bench uses one head dimension for keys and "
                f"values; this shape has dqk={dqk}, dv={dv}"
            )
        return None

    def _tokens(self) -> int:
        """Prefill sequence length, or decode tokens per sequence per step."""
        if self._kernel == "kda_attention_prefill":
            return self._seq_len
        return int(self._shape.get("tokens_per_seq", 1))

    def build_args(self) -> str:
        shape = self._shape
        common = (
            f"--batch_sizes {self._batch_size} --seq_lens {self._tokens()} "
            f"--num_heads {shape['hq']} --backends gluon"
        )
        if self._kernel == "kda_attention_prefill":
            metric = "throughput" if self._metric == "bandwidth" else self._metric
            return f"{common} --metric {metric} --forward-only"
        return (
            f"{common} --head_dim {shape['dqk']} -metric {self._metric} "
            "--gate --paged --state_dtype fp32 --dtype bf16 --no-spread"
        )

    def parse_stdout(self, stdout: str) -> float:
        result = _last_number(stdout, "KDA")
        if self._metric == "time":
            return result * 1e-3  # us -> ms
        return result * 1e3  # TB/s -> GB/s

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._tokens(),
            "hq": shape["hq"],
            "dqk": shape["dqk"],
            "dv": shape["dv"],
            self._metric: bench_result,
        }


class AttnResKernelHandler(KernelHandler):
    """Handler for Kimi attention-residual benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        return [shape.copy() for shape in shapes]

    def build_args(self) -> str:
        shape = self._shape
        op = str(shape.get("op", "gate"))
        args = (
            f"-N {self._M} -D {shape['D']} -L {shape['L']} "
            f"--op {op} --metric {self._metric} --dtype bf16"
        )
        if op == "fwd":
            args += f" --layout {shape.get('layout', 'packed')}"
        for field, flag in (
            ("add_hidden", "--add-hidden"),
            ("add_hidden2", "--add-hidden2"),
            ("close_block", "--close-block"),
            ("onorm", "--onorm"),
        ):
            if shape.get(field):
                args += f" {flag}"
        return args

    def parse_stdout(self, stdout: str) -> float:
        return _last_number(stdout, "attn_res")

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "M": self._M,
            "D": shape["D"],
            "L": shape["L"],
            "attn_res_op": shape.get("op", "gate"),
            self._metric: bench_result,
        }


class GatedDeltaNetKernelHandler(KernelHandler):
    """Handler for fused Qwen Gated DeltaNet prefill and decode benchmarks."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["hk"])
            result.append(s)
        return result

    def unsupported_reason(self, shape: ShapeDict) -> str | None:
        hk = int(shape["hk"])
        hv = hk * int(shape.get("ratio", 2))
        dqk = int(shape["dqk"])
        dv = int(shape.get("dv", dqk))
        if self._kernel == "gated_delta_net_decode":
            if dv == dqk:
                return None
            return (
                "the decode bench uses one head dimension for keys and values; "
                f"this shape has dqk={dqk}, dv={dv}"
            )
        supported = (
            gdn_prefill._K_HEADS,
            gdn_prefill._V_HEADS,
            gdn_prefill._HEAD_DIM,
            gdn_prefill._HEAD_DIM,
        )
        if (hk, hv, dqk, dv) == supported:
            return None
        return (
            "the fused prefill kernel supports only "
            f"hk={supported[0]}, hv={supported[1]}, dqk=dv={supported[2]} per GPU; "
            f"this shape has hk={hk}, hv={hv}, dqk={dqk}, dv={dv} at TP={self._tp}"
        )

    def build_args(self) -> str:
        shape = self._shape
        common = f"--num-k-heads {shape['hk']} --head-dim {shape['dqk']}"
        if self._kernel == "gated_delta_net_prefill":
            return f"--shapes {self._batch_size}x{self._seq_len} {common}"

        args = (
            f"--batch {self._batch_size} {common} "
            f"--ratio {shape.get('ratio', 2)} "
            f"--layout {shape.get('layout', 'flat')} "
            f"--slots {shape.get('slots', 512)} "
            f"--state-dtype {shape.get('state_dtype', 'fp32')}"
        )
        if not shape.get("fp8", False):
            args += " --no-fp8"
        return args

    def parse_stdout(self, stdout: str) -> float:
        lines = [line for line in stdout.splitlines() if line.strip()]
        is_prefill = self._kernel == "gated_delta_net_prefill"
        if is_prefill and "(uncovered)" in lines[-1]:
            raise ValueError(
                "The fused prefill kernel has no tile for "
                f"batch={self._batch_size}, tokens={self._M}. It covers "
                f"{gdn_prefill._MIN_TOKENS}-{gdn_prefill._MAX_TOKENS} tokens "
                f"and batch <= {gdn_prefill._MAX_BATCH}."
            )
        values = lines[-1].split()
        # Prefill rows: batch, seqlen, tokens, tile, us, Mtok/s, TB/s, %peak.
        # Decode rows: batch, us, GB moved, TB/s, %peak.
        if len(values) != (8 if is_prefill else 5):
            raise ValueError(f"Unexpected Gated DeltaNet bench output: {lines[-1]!r}")
        try:
            if self._metric == "time":
                return float(values[-4]) * 1e-3  # us -> ms
            return float(values[-2]) * 1e3  # TB/s -> GB/s
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Unexpected Gated DeltaNet bench output: {lines[-1]!r}"
            ) from error

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        ratio = int(shape.get("ratio", 2))
        is_decode = self._kernel == "gated_delta_net_decode"
        row: ResultRow = {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": 1 if is_decode else self._seq_len,
            "hk": shape["hk"],
            "hv": int(shape["hk"]) * ratio,
            "dqk": shape["dqk"],
            "dv": shape.get("dv", shape["dqk"]),
        }
        # The prefill kernel has one fixed (interleaved) QKVZ layout.
        if is_decode:
            row["layout"] = shape.get("layout", "flat")
        row[self._metric] = bench_result
        return row


class UnifiedAttnKernelHandler(KernelHandler):
    """Handler for unified attention benchmarks (bench_unified_attention.py)."""

    def get_tp_shapes(self, shapes: list[ShapeDict]) -> list[ShapeDict]:
        result = []
        for shape in shapes:
            s = shape.copy()
            self._shard_keys(s, ["hq", "hkv"])
            result.append(s)
        return result

    def build_args(self) -> str:
        shape = self._shape
        sliding_window = shape.get("sliding_window", None)
        args = (
            f"-b {self._batch_size} -hq {shape['hq']} -hk {shape['hkv']} "
            f"-d {shape['dqk']} -dv {shape['dv']} -sq {self._seq_len} -sk {self._seq_len} "
            f"--metric {self._metric}"
        )
        # A shape without block_size uses bench_unified_attention's own default.
        block_size = shape.get("block_size")
        if block_size is not None:
            args += f" -block_size {int(block_size)}"
        if sliding_window is not None:
            args += f" -sliding_window {sliding_window}"
        if shape.get("output_gate", False):
            args += " -output_gate"
        return args

    def parse_stdout(self, stdout: str) -> float:
        lines = [line.split() for line in stdout.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(
                f"Unexpected unified_attention bench output: expected at least 3 lines, got {len(lines)}"
            )
        data = lines[2]
        if len(data) < 10:
            raise ValueError(f"Unexpected unified_attention bench data line: {data!r}")
        return float(data[9])

    def build_result_row(self, bench_result: float | str) -> ResultRow:
        shape = self._shape
        block_size = shape.get("block_size")
        return {
            "Model": self._model,
            "Kernel": self._kernel,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "hq": shape["hq"],
            "hkv": shape["hkv"],
            "dqk": shape["dqk"],
            "dv": shape["dv"],
            "sliding_window": shape.get("sliding_window", None),
            "block_size": (
                int(block_size) if block_size is not None else DEFAULT_BLOCK_SIZE
            ),
            "output_gate": shape.get("output_gate", False),
            self._metric: bench_result,
        }


_HANDLER_RULES: list[tuple[Callable[[str], bool], type[KernelHandler]]] = [
    (lambda k: "moe" in k, MoeKernelHandler),
    (lambda k: "gemm" in k and "moe" not in k, GemmKernelHandler),
    (lambda k: k == "rmsnorm", RmsnormKernelHandler),
    (lambda k: k == "fused_rms_mxfp4_quant", FusedRmsMxfp4QuantKernelHandler),
    (lambda k: k == "rope", RopeKernelHandler),
    (lambda k: k == "mha", MhaKernelHandler),
    (lambda k: k == "mla", MlaKernelHandler),
    (lambda k: k == "mhc", MhcKernelHandler),
    (lambda k: k.startswith("kda_attention_"), KdaAttentionKernelHandler),
    (lambda k: k == "attn_res", AttnResKernelHandler),
    (lambda k: k.startswith("gated_delta_net_"), GatedDeltaNetKernelHandler),
    (lambda k: k == "sparse_mqa", SparseMqaKernelHandler),
    (lambda k: k == "unified_attention", UnifiedAttnKernelHandler),
]


def _get_handler(kernel: str) -> KernelHandler:
    for rule, handler in _HANDLER_RULES:
        if rule(kernel):
            return handler()
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


def validate_result(value: float | str, handler: KernelHandler) -> float | str:
    # Time, throughput, and bandwidth are always positive. Zero or negative is
    # considered sentinel value.
    try:
        number = float(value)
    except (TypeError, ValueError):
        print(f"Non-numeric result on {handler.to_str()}: {value!r}")
        return "N/A"
    if not math.isfinite(number) or number <= 0:
        print(f"Invalid result on {handler.to_str()}: {value!r}")
        return "N/A"
    return value


def call_function(
    bench_fn: Callable[[list[str]], None], handler: KernelHandler
) -> float | str:
    stdout = io.StringIO()
    stderr = io.StringIO()
    result: float | str = "N/A"

    try:
        with (
            warnings.catch_warnings(record=True) as captured_warnings,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            bench_fn(shlex.split(handler.build_args()))
        warning_output = "".join(
            warnings.formatwarning(
                message=w.message,
                category=w.category,
                filename=w.filename,
                lineno=w.lineno,
                line=w.line,
            )
            for w in captured_warnings
        )
        if warning_output:
            # TODO: Replace this print statement with proper logging - warning.
            print(f"Warnings stream isn't empty: [{warning_output}]")
        if stderr.getvalue():
            # TODO: Replace this print statement with proper logging - error.
            print(f"Standard error stream isn't empty: [{stderr.getvalue()}]")
        else:
            result = validate_result(handler.parse_stdout(stdout.getvalue()), handler)
    except OutOfResources as e:
        # Parse the error message to extract required LDS and hardware limit.
        # Expected format: "out of resource: shared memory, Required: XXXX, Hardware limit: XXXX..."
        match = re.search(r"Required:\s*(\d+),\s*Hardware limit:\s*(\d+)", str(e))
        if match:
            required = int(match.group(1))
            hw_limit = int(match.group(2))
            ratio: float = required / hw_limit
            print(
                f"Out of LDS on {handler.to_str()}: {required} / {hw_limit} "
                f"({ratio:.1f}x)"
            )
        else:
            print(f"Out of resources while benchmarking {handler.to_str()}. {e}")

    except (Exception, SystemExit) as e:  # noqa: BLE001
        print(
            f"Unexpected error while benchmarking {handler.to_str()}. {type(e).__name__}: {e}"
        )

    # Close matplotlib figures to silence errors and avoid memory leaks.
    plt.close("all")
    return result


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

            # FP8xFP8 path, verified on gfx950 and gfx1250.
            # gfx942: FNUZ tensors, USE_FNUZ=False.
            # gfx1200/gfx1201: RDNA; this script is CDNA only.
            if kernel == "moe_op_gemm_a8w8" and arch_info.get_arch() not in (
                "gfx950",
                "gfx1250",
            ):
                print(
                    f"moe_op_gemm_a8w8 is not verified on this device. Skipping {kernel}."
                )
                continue

            # bench_mla_decode.py hardcodes the DeepSeek-V3 MLA dimensions and
            # takes only the head count from model_configs.json. Run it only for
            # models with a model_configs.json entry in MLA_BENCH_MODELS.
            if kernel == "mla" and model not in MLA_BENCH_MODELS:
                print(
                    f"Skipping mla for model '{model}': "
                    "bench_mla_decode.py has no config for this model."
                )
                continue

            if kernel == "mhc" and metric == "throughput":
                print(
                    "Skipping mhc: throughput is not supported; "
                    "use --metric time or bandwidth."
                )
                continue

            if kernel.startswith("kda_attention_"):
                if arch_info.get_arch() not in ("gfx950", "gfx1250"):
                    print(f"KDA is not supported on this device. Skipping {kernel}.")
                    continue
                if metric == "throughput":
                    print(
                        f"Skipping {kernel}: TFLOPS throughput is not supported; "
                        "use --metric time or bandwidth."
                    )
                    continue

            if kernel == "attn_res" and metric == "throughput":
                print(
                    "Skipping attn_res: throughput is not supported; "
                    "use --metric time or bandwidth."
                )
                continue

            if kernel.startswith("gated_delta_net_"):
                if arch_info.get_arch() != "gfx950":
                    print(
                        "Fused Gated DeltaNet is not supported on this device. "
                        f"Skipping {kernel}."
                    )
                    continue
                if metric == "throughput":
                    print(
                        f"Skipping {kernel}: TFLOPS throughput is not supported; "
                        "use --metric time or bandwidth."
                    )
                    continue

            # MLA only reports time (ms)
            run_metric = "time" if kernel == "mla" else metric

            bench_fn = KERNEL_DICT[kernel]
            handler = _get_handler(kernel)

            handler.set_run(model, kernel, run_metric, gemm_layout, mha_layout, TP)
            tp_shapes = handler.get_tp_shapes(shapes)
            kernel_seq_lens = (
                seq_lens[:1] if kernel in DECODE_ONLY_KERNELS else seq_lens
            )
            for shape in tp_shapes:
                reason = handler.unsupported_reason(shape)
                if reason is not None:
                    print(f"Skipping {kernel} for model '{model}': {reason}.")
                    continue
                for batch_size in batch_sizes:
                    for seq_len in kernel_seq_lens:
                        handler.set_iteration(shape, batch_size, seq_len)
                        bench_result = call_function(bench_fn, handler)
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


def parse_args(
    available_models: list[str], available_kernels: list[str]
) -> argparse.Namespace:
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
            "Model name filter: case-insensitive regex matched against model names "
            "(default: all models). "
            "e.g. 'llama3' to include only Llama3 family, "
            "'llama|qwen' to include both Llama and Qwen families, "
            "'^(?!.*deepseek)' to exclude DeepSeek family."
            f"\nAvailable models: {', '.join(available_models)}."
        ),
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help=(
            "Kernel name filter: case-insensitive regex matched against kernel names "
            "(default: all kernels). "
            "e.g. 'gemm' to include any kernel name containing gemm, "
            "'moe|rmsnorm' for MoE and RMSNorm."
            f"\nAvailable kernels: {', '.join(available_kernels)}."
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


def filter_models_and_kernels(
    data: ModelShapesData,
    available_models: list[str],
    model_pattern: str | None,
    kernel_pattern: str | None,
) -> ModelShapesData | None:

    def _filter_by_regex(
        pattern: str, pattern_name: str, candidates: list[str]
    ) -> list[str]:
        try:
            pat = re.compile(pattern, re.IGNORECASE)
        except re.error:
            print(
                f"Invalid {pattern_name} regex: {pattern!r} - running all {pattern_name}s."
            )
            return candidates
        return [n for n in candidates if pat.search(n) is not None]

    if model_pattern is not None:
        matched_models = _filter_by_regex(model_pattern, "model", available_models)
        data = {m: data[m] for m in matched_models}
        if not data:
            print("There are no models after filtering by model name.")
            return None

    if kernel_pattern is not None:
        filtered: ModelShapesData = {}
        for m, kernels in data.items():
            matched_kernels = _filter_by_regex(
                kernel_pattern, "kernel", sorted(kernels.keys())
            )
            kept = {k: kernels[k] for k in matched_kernels}
            if kept:
                filtered[m] = kept
        data = filtered
        if not data:
            print("There are no models/kernels after filtering by kernel name.")
            return None

    return data


def main() -> None:
    data = read_json("model_shapes.json")
    available_models = sorted(data.keys())
    available_kernels = sorted(KERNEL_DICT.keys())
    args = parse_args(available_models, available_kernels)

    models = args.model
    kernels = args.kernel
    batch_sizes = args.batch_size
    seq_lens = args.seq_len
    TP = args.TP
    metric = args.metric
    gemm_layout = args.gemm_layout
    mha_layout = args.mha_layout
    output_file = args.output_file

    filtered_data = filter_models_and_kernels(data, available_models, models, kernels)
    if filtered_data is None:
        return
    data = filtered_data

    results = run_benchmarks(
        data, batch_sizes, seq_lens, TP, gemm_layout, mha_layout, metric
    )

    print_and_save_results(results, metric, output_file)


if __name__ == "__main__":
    main()
