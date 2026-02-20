import argparse
import csv
import io
import logging
import os
import re
import shlex
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal, Optional, Self, get_args

import matplotlib.pyplot as plt
from triton import next_power_of_2
from triton.runtime.errors import OutOfResources


def disable_logs(logger: str) -> None:
    logging.getLogger(logger).disabled = True
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


disable_logs("aiter")
from bench_mha import main as bench_mha_main  # noqa: E402
from bench_mla_decode import main as bench_mla_main  # noqa: E402

# Module-level tracking for head dimension warnings
# Stores (model_name, kernel) tuples to ensure each warning is logged once
_logged_hdim_warnings: set[tuple[str, str]] = set()

# Default benchmark parameter values
DEFAULT_BATCH_START: int = 1
DEFAULT_BATCH_INC: int = 1
DEFAULT_BATCH_END: int = 8
DEFAULT_SEQ_START: int = 1024
DEFAULT_SEQ_INC: int = 1024
DEFAULT_SEQ_END: int = 8192


@dataclass(kw_only=True)
class Model:
    name: str
    hq: int
    hkv: int
    dqk: int
    dv: int
    use_mla: bool = False

    def __post_init__(self) -> None:
        assert self.name, "Model name must be non-empty."
        assert self.hq > 0, "Number of query heads must be positive."
        assert self.hkv > 0, "Number of key and value heads must be positive."
        assert (
            self.hq % self.hkv == 0
        ), f"Number of query heads ({self.hq}) must be divisible by number of key-value heads ({self.hkv})."
        assert self.dqk > 0, "Dimension of query and key heads must be positive."
        assert self.dv > 0, "Dimension of value heads must be positive."
        assert (
            self.dqk >= self.dv
        ), f"Invalid head dimensions: dqk ({self.dqk}) < dv ({self.dv}). Expected dqk >= dv."

    def kernel_backend_str(self) -> str:
        return "mla" if self.use_mla else "mha"

    def effective_d_qk_v(self, kernel: "Kernel") -> tuple[int, int]:
        """
        Compute effective head dimensions for the given kernel.

        Forward kernels support arbitrary head dimensions.
        Backward kernels require power of 2 head dimensions:
        - bwdo (one kernel):
          * If dqk == dv, promote to next power of 2
          * If dqk > dv, ensure both dv and d_pe = dqk - dv are powers of 2
        - bwdf (fused):
          * If dqk == dv, promote to next power of 2
          * If dqk > dv, doesn't support PE, use next_power_of_2(dqk) for both

        Args:
            kernel: The kernel type

        Returns:
            Tuple of (effective_dqk, effective_dv)
        """
        if kernel == "fwd":
            return (self.dqk, self.dv)

        # Backward kernels require power of 2 head dimensions
        effective_dqk: int
        effective_dv: int

        if self.dqk == self.dv:
            # Case 1: dqk == dv == d
            # If not a power of 2, promote to next power of 2
            # (same logic for both bwdo and bwdf)
            effective_dqk = effective_dv = next_power_of_2(self.dqk)
        else:
            # Case 2: dqk > dv (guaranteed by __post_init__ assertion)
            if kernel == "bwdo":
                # bwdo: d_pe = dqk - dv, and both dv and d_pe must be powers of 2
                d_pe: int = self.dqk - self.dv
                effective_dv = next_power_of_2(self.dv)
                effective_d_pe: int = next_power_of_2(d_pe)
                effective_dqk = effective_dv + effective_d_pe
            else:
                # bwdf: doesn't support PE, use next power of 2 of QK dim for both
                effective_dqk = effective_dv = next_power_of_2(self.dqk)

        # Log warning once per unique (model_name, kernel) if dimensions changed
        if effective_dqk != self.dqk or effective_dv != self.dv:
            warning_key: tuple[str, str] = (self.name, kernel)
            if warning_key not in _logged_hdim_warnings:
                if kernel == "bwdo":
                    logging.warning(
                        "%s: Effective head sizes aren't equal to the original values. "
                        "Backward one-kernel only supports power of 2 head sizes. "
                        "dqk: %d -> %d, dv: %d -> %d",
                        self.name,
                        self.dqk,
                        effective_dqk,
                        self.dv,
                        effective_dv,
                    )
                elif kernel == "bwdf":
                    logging.warning(
                        "%s: Effective head sizes aren't equal to the original values. "
                        "Backward fused only supports power of 2 head sizes without PE. "
                        "dqk: %d -> %d, dv: %d -> %d",
                        self.name,
                        self.dqk,
                        effective_dqk,
                        self.dv,
                        effective_dv,
                    )
                _logged_hdim_warnings.add(warning_key)

        return (effective_dqk, effective_dv)

    @classmethod
    def new(cls, name: str) -> "ModelBuilder":
        return ModelBuilder(name)


class ModelBuilder:
    name: str
    hq: int
    hkv: int
    dqk: int
    dv: int
    use_mla: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

    def h(self, h: int) -> Self:
        self.hq = self.hkv = h
        return self

    def h_q_vk(self, hq: int, hkv: int) -> Self:
        self.hq = hq
        self.hkv = hkv
        return self

    def d(self, d: int) -> Self:
        self.dqk = self.dv = d
        return self

    def d_qk_v(self, dqk: int, dv: int) -> Self:
        self.dqk = dqk
        self.dv = dv
        return self

    def mla(self) -> Self:
        self.use_mla = True
        return self

    def build(self) -> Model:
        return Model(
            name=self.name,
            hq=self.hq,
            hkv=self.hkv,
            dqk=self.dqk,
            dv=self.dv,
            use_mla=self.use_mla,
        )


TpDegree = Literal[1, 2, 4, 8]


@dataclass(kw_only=True)
class TpModel:
    model: Model
    tp: TpDegree = 1

    def __post_init__(self) -> None:
        assert self.tp > 0, "Tensor parallelism must be positive."
        assert (
            self.model.hq % self.tp == 0
        ), "Number of query heads must be divisible by tensor parallelism."

        original_model: Model = self.model
        self.model = Model(
            name=original_model.name,
            hq=original_model.hq // self.tp,
            hkv=max(original_model.hkv // self.tp, 1),
            dqk=original_model.dqk,
            dv=original_model.dv,
            use_mla=original_model.use_mla,
        )


# There are two backward implementations:
# * "one kernel", the default one, referred as "bwdo"
# * "fused", the legacy one, referred as "bwdf"
Kernel = Literal["fwd", "bwdo", "bwdf"]


Layout = Literal["bshd", "thd"]


@dataclass(kw_only=True, frozen=True)
class Metric:
    """Represents a benchmark metric with its name and unit."""

    name: str
    unit: str
    user_unit: str

    def __post_init__(self) -> None:
        assert self.name, "Metric name must be non-empty."
        assert self.unit, "Metric unit must be non-empty."
        assert self.user_unit, "Metric user facing unit must be non-empty."


# Available benchmark metrics:
METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in [
        Metric(name="time", unit="ms", user_unit="ms"),
        Metric(name="throughput", unit="tflops", user_unit="TFLOPS"),
        Metric(name="bandwidth", unit="gpbs", user_unit="GB/s"),
    ]
}


@dataclass(kw_only=True)
class BenchArgs:
    kernel: Kernel
    layout: Optional[Layout]
    tp_model: TpModel
    b: int
    s: int

    def __post_init__(self) -> None:
        assert self.tp_model.model.use_mla == (
            self.layout is None
        ), "Layout must be absent for MLA backed models or present for MHA backed models."
        assert self.b > 0, "Batch size must be positive."
        assert self.s > 0, "Sequence length must be positive."

    def to_mha_cli_str(self, metric: Metric) -> str:
        """Convert to CLI string of `bench_mha.py`."""
        assert self.layout is not None, "Layout must be present to run MHA benchmark."

        m: Model = self.tp_model.model
        s: str = str(self.s)

        effective_dqk: int
        effective_dv: int
        effective_dqk, effective_dv = m.effective_d_qk_v(self.kernel)

        args_dict: dict[str, str] = {
            "-mode": self.kernel[:3],
            "-causal": "true",
            "--layout": self.layout,
            "--dtype": "bf16",
            "-b": str(self.b),
            "-hq": str(m.hq),
            "-hk": str(m.hkv),
            "-sq": s,
            "-sk": s,
            "-d": str(effective_dqk),
            "-dv": str(effective_dv),
            "-metric": metric.name,
        }

        args_list: list[str] = [kv for k, v in args_dict.items() for kv in (k, v)]
        if self.kernel == "bwdf":
            args_list.append("-fused_bwd")
        args_str: str = " ".join(args_list)

        return args_str

    def to_mla_cli_str(self) -> str:
        """Convert to CLI string of `bench_mla_decode.py`."""
        assert self.kernel == "fwd", "MLA only support forward kernel."

        args_dict: dict[str, str] = {
            "--model": "deepseek-V3",
            "--dtype": "bf16",
            "-b": str(self.b),
            "--seqlen": str(self.s),
        }

        args_list: list[str] = [kv for k, v in args_dict.items() for kv in (k, v)]
        args_list.extend(("-equal_seqlens", "-causal"))

        args_str: str = " ".join(args_list)
        return args_str

    def to_log_str(self) -> str:
        """Convert to log string."""
        m: Model = self.tp_model.model
        log_dict: dict[str, str] = {
            "kernel_backend": m.kernel_backend_str(),
            "kernel": self.kernel,
            "layout": str(self.layout),
            "model": m.name,
            "hq": str(m.hq),
            "hkv": str(m.hkv),
            "dqk": str(m.dqk),
            "dv": str(m.dv),
            "tp": str(self.tp_model.tp),
            "b": str(self.b),
            "s": str(self.s),
        }
        log_str: str = ", ".join(f"{k}={v}" for k, v in log_dict.items())
        return f"({log_str})"

    @classmethod
    def csv_header(cls, metric: Metric) -> list[str]:
        """Return CSV header as a list of strings."""
        return [
            "kernel_backend",
            "kernel",
            "layout",
            "model",
            "hq",
            "hkv",
            "dqk",
            "dv",
            "tp",
            "b",
            "s",
            metric.unit,
        ]

    def csv_data(self, perf: Optional[float] = None) -> list[str | int | float | None]:
        """Return CSV data row as a list of mixed types."""
        m: Model = self.tp_model.model
        return [
            m.kernel_backend_str(),
            self.kernel,
            self.layout,
            m.name,
            m.hq,
            m.hkv,
            m.dqk,
            m.dv,
            self.tp_model.tp,
            self.b,
            self.s,
            perf,
        ]


def get_stdout(out: str, err: str, num_out_lines: int) -> Optional[list[list[str]]]:
    assert num_out_lines >= 0, "Expected number of stdout lines must be non-negative."
    # Check empty stderr:
    if err:
        logging.error("Standard error stream isn't empty: [%s]", err)
        return None
    # Split stdout:
    out_lines: list[list[str]] = [
        out_line.split() for out_line in out.strip().split(sep="\n")
    ]
    # Check number of lines in stdout:
    if len(out_lines) != num_out_lines:
        logging.error(
            "Standard out stream doesn't have %d lines: [%s]", num_out_lines, out
        )
        return None
    return out_lines


def get_mha_bench_result(
    args: BenchArgs, metric: Metric, out: str, err: str
) -> Optional[float]:
    """Get result from `bench_mha.py`."""
    # Get preprocessed stdout:
    out_lines: Optional[list[list[str]]] = get_stdout(out, err, num_out_lines=3)
    if out_lines is None:
        return None
    l0: list[str]
    l1: list[str]
    l2: list[str]
    l0, l1, l2 = out_lines
    # Check stdout line #1 (benchmark name):
    if l0 != ["bench_mha:"]:
        logging.error("Benchmark name doesn't match: %s", l0)
        return None
    # Check stdout line #2 (table header):
    kernel_header: str = {"fwd": "fwd", "bwdo": "onekernel-bwd", "bwdf": "fused-bwd"}[
        args.kernel
    ]
    if l1 != [
        "BATCH",
        "HQ",
        "HK",
        "N_CTX_Q",
        "N_CTX_K",
        f"{kernel_header}({metric.user_unit})",
        f"({metric.user_unit})",
    ]:
        logging.error("Table header doesn't match: %s", l1)
        return None
    # Check stdout line #3 (table data):
    m: Model = args.tp_model.model
    try:
        if not all(
            [
                len(l2) == 7,
                l2[0] == "0",
                int(float(l2[1])) == args.b,
                int(float(l2[2])) == m.hq,
                int(float(l2[3])) == m.hkv,
                int(float(l2[4])) == args.s,
                int(float(l2[5])) == args.s,
            ]
        ):
            logging.error("Table data doesn't match: %s", l2)
            return None
        return float(l2[6])
    except ValueError as e:
        logging.error(
            "Unexpected numeric conversion error. %s: %s", type(e).__name__, e
        )
        return None


def get_mla_bench_result(args: BenchArgs, out: str, err: str) -> Optional[float]:
    """Get result from `bench_mla_decode.py`."""
    # Get preprocessed stdout:
    out_lines: Optional[list[list[str]]] = get_stdout(out, err, num_out_lines=3)
    if out_lines is None:
        return None
    l0: list[str]
    l1: list[str]
    l2: list[str]
    l0, l1, l2 = out_lines
    # Check stdout line #1 (benchmark name):
    if l0 != ["bench_mla_decode:"]:
        logging.error("Benchmark name doesn't match: %s", l0)
        return None
    # Check stdout line #2 (table header):
    if l1 != [
        "model",
        "B",
        "H",
        "S",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "rotary_dim",
        "num_kv_splits",
        "mla_decode_fwd",
        "(ms)",
    ]:
        logging.error("Table header doesn't match: %s", l1)
        return None
    # Check stdout line #3 (table data):
    try:
        if not all(
            [
                len(l2) == 10,
                l2[0] == "0",
                l2[1] == "deepseek-V3",
                int(float(l2[2])) == args.b,
                int(float(l2[3])) == args.tp_model.model.hq,
                int(float(l2[4])) == args.s,
                int(float(l2[5])) == 512,
                int(float(l2[6])) == 64,
                int(float(l2[7])) == 64,
                # TODO: Evaluate other `num_kv_splits` values, according to TP degree.
                int(float(l2[8])) == 32,
            ]
        ):
            logging.error("Table data doesn't match: %s", l2)
            return None
        return float(l2[9])
    except ValueError as e:
        logging.error(
            "Unexpected numeric conversion error. %s: %s", type(e).__name__, e
        )
        return None


def run_bench(args: BenchArgs, metric: Metric) -> Optional[float]:
    perf: Optional[float] = None

    out = io.StringIO()
    err = io.StringIO()

    try:
        if args.tp_model.model.use_mla:
            assert (
                metric == METRICS["time"]
            ), "MLA benchmark only generates performance in milliseconds."
            with redirect_stdout(out), redirect_stderr(err):
                bench_mla_main(shlex.split(args.to_mla_cli_str()))
            perf = get_mla_bench_result(args, out.getvalue(), err.getvalue())
        else:
            with redirect_stdout(out), redirect_stderr(err):
                bench_mha_main(shlex.split(args.to_mha_cli_str(metric)))
            perf = get_mha_bench_result(args, metric, out.getvalue(), err.getvalue())

    except OutOfResources as e:
        # Parse the error message to extract required LDS and hardware limit.
        # Expected format: "out of resource: shared memory, Required: XXXX, Hardware limit: XXXX..."
        match = re.search(r"Required:\s*(\d+),\s*Hardware limit:\s*(\d+)", str(e))
        if match:
            required = int(match.group(1))
            hw_limit = int(match.group(2))
            ratio: float = required / hw_limit
            logging.error(
                "Out of LDS on %s: %d / %d (%.1fx)",
                args.to_log_str(),
                required,
                hw_limit,
                ratio,
            )
        else:
            logging.error(
                "Out of resources while benchmarking %s. %s", args.to_log_str(), e
            )

    except Exception as e:
        logging.error(
            "Unexpected error while benchmarking %s. %s: %s",
            args.to_log_str(),
            type(e).__name__,
            e,
        )

    finally:
        # Close matplotlib figures to silence errors and avoid memory leaks.
        plt.close("all")

    return perf


def get_models(model_filter: Optional[str] = None) -> list[Model]:
    all_models: list[Model] = [
        Model.new("Llama3 405B").h_q_vk(128, 8).d(128).build(),
        Model.new("Llama3 70B").h_q_vk(64, 8).d(128).build(),
        Model.new("Llama3 8B").h_q_vk(32, 8).d(128).build(),
        Model.new("Llama4 Maverick (Text)").h_q_vk(40, 8).d(128).build(),
        Model.new("Llama4 Maverick (Vision)").h(16).d(88).build(),
        Model.new("Qwen-235B-A22B").h_q_vk(64, 4).d(128).build(),
        Model.new("GPT-OSS 120B").h_q_vk(64, 8).d(64).build(),
        Model.new("DeepSeek V3 (Prefill)").h(128).d_qk_v(192, 128).build(),
        Model.new("DeepSeek V3 (Decode)").mla().h_q_vk(128, 1).d_qk_v(576, 512).build(),
    ]
    model_names: list[str] = [model.name for model in all_models]
    assert len(model_names) == len(
        set(model_names)
    ), "Duplicate model names found. Model names must be unique."

    if model_filter is None:
        return all_models  # model_filter is None, return all

    model_filter = model_filter.strip()
    if not model_filter:  # Empty string after stripping
        logging.debug("Empty model name filter, returning all models.")
        return all_models

    try:
        pattern: re.Pattern[str] = re.compile(model_filter, re.IGNORECASE)
    except re.error:
        logging.warning(
            "Invalid model filter regex: %r - returning all models.",
            model_filter,
        )
        return all_models

    filtered_models: list[Model] = [
        model for model in all_models if pattern.search(model.name)
    ]
    logging.debug("Number of filtered models: %d", len(filtered_models))
    if not filtered_models:
        logging.warning("There are no models after filtering by model name.")
    return filtered_models


def list_models() -> None:
    """Log all available models with head counts and dimensions."""
    logging.info("Available models:")
    for model in get_models():
        logging.info(
            "%s kernel_backend=%s hq=%d hkv=%d dqk=%d dv=%d",
            model.name,
            model.kernel_backend_str(),
            model.hq,
            model.hkv,
            model.dqk,
            model.dv,
        )


def get_tp_models(
    models: list[Model] = get_models(),
    tps: Iterable[TpDegree] = get_args(TpDegree),
) -> list[TpModel]:
    return [TpModel(model=model, tp=tp) for model, tp in product(models, tps)]


@dataclass(kw_only=True)
class Range:
    start: int
    inc: int
    end: int

    def __post_init__(self) -> None:
        assert self.start > 0, "Start must be positive."
        assert self.inc > 0, "Increment must be positive."
        assert self.end > 0, "End must be positive."
        assert self.end >= self.start, "End must be greater than or equal to start."

    def to_range(self) -> range:
        return range(self.start, self.end + 1, self.inc)


def get_bench_args(
    kernels: Iterable[Kernel] = get_args(Kernel),
    layouts: Iterable[Layout] = get_args(Layout),
    tp_models: list[TpModel] = get_tp_models(),
    batch_range: Range = Range(
        start=DEFAULT_BATCH_START, inc=DEFAULT_BATCH_INC, end=DEFAULT_BATCH_END
    ),
    seq_range: Range = Range(
        start=DEFAULT_SEQ_START, inc=DEFAULT_SEQ_INC, end=DEFAULT_SEQ_END
    ),
) -> list[BenchArgs]:
    # MHA kernel backend:
    bench_args: list[BenchArgs] = [
        BenchArgs(kernel=kernel, layout=layout, tp_model=tp_model, b=b, s=s)
        for kernel, layout, tp_model, b, s in product(
            kernels,
            layouts,
            (tp_model for tp_model in tp_models if not tp_model.model.use_mla),
            batch_range.to_range(),
            seq_range.to_range(),
        )
    ]
    # MLA kernel backend:
    # Only forward kernel, layout option doesn't make sense.
    if "fwd" in kernels:
        bench_args.extend(
            [
                BenchArgs(kernel="fwd", layout=None, tp_model=tp_model, b=b, s=s)
                for tp_model, b, s in product(
                    (tp_model for tp_model in tp_models if tp_model.model.use_mla),
                    batch_range.to_range(),
                    seq_range.to_range(),
                )
            ]
        )
    return bench_args


class Stats:
    """Tracks benchmark statistics including total count and failures."""

    num_benchmarks: int
    num_failures: int

    def __init__(self) -> None:
        self.num_benchmarks = 0
        self.num_failures = 0

    def report_success(self) -> None:
        self.num_benchmarks += 1

    def report_failure(self) -> None:
        self.num_benchmarks += 1
        self.num_failures += 1

    def failure_percentage(self) -> float:
        return (
            0.0
            if self.num_benchmarks == 0
            else (self.num_failures / self.num_benchmarks) * 100.0
        )


class GlobalStats:
    """Tracks global statistics and per-(kernel, model) statistics."""

    global_stats: Stats
    kernel_model_stats: dict[tuple[str, str], Stats]

    def __init__(self) -> None:
        self.global_stats = Stats()
        self.kernel_model_stats = {}

    def _get_or_create_stats(self, kernel: str, model: str) -> Stats:
        """Get or lazily create stats for a (kernel, model) pair."""
        key: tuple[str, str] = (kernel, model)
        if key not in self.kernel_model_stats:
            self.kernel_model_stats[key] = Stats()
        return self.kernel_model_stats[key]

    def report_success(self, kernel: str, model: str) -> None:
        self.global_stats.report_success()
        self._get_or_create_stats(kernel, model).report_success()

    def report_failure(self, kernel: str, model: str) -> None:
        self.global_stats.report_failure()
        self._get_or_create_stats(kernel, model).report_failure()

    def log_stats(self) -> None:
        """Log aggregated statistics about benchmark failures."""
        # Early exit if no failures
        if self.global_stats.num_failures == 0:
            return

        # Overall failure statistics
        logging.info("=== Benchmark Statistics ===")
        logging.info(
            "Total failures: %d / %d (%.2f%%)",
            self.global_stats.num_failures,
            self.global_stats.num_benchmarks,
            self.global_stats.failure_percentage(),
        )

        # Failures grouped by kernel and model
        logging.info("=== Failures by Kernel and Model ===")
        for (kernel, model), stats in sorted(self.kernel_model_stats.items()):
            if stats.num_failures > 0:
                logging.info(
                    "[%s, %s]: %d / %d failures (%.2f%%)",
                    kernel,
                    model,
                    stats.num_failures,
                    stats.num_benchmarks,
                    stats.failure_percentage(),
                )


def positive_int(value: str) -> int:
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not an integer")
    if int_value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return int_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark attention kernels with configurations of popular LLM models.",
        add_help=True,
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str.lower,
        nargs="+",
        choices=get_args(Kernel),
        default=get_args(Kernel),
        help="attention kernels (default: all)",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str.lower,
        nargs="+",
        choices=get_args(Layout),
        default=get_args(Layout),
        help="memory layouts (default: all)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "model name filter: case-insensitive regex matched against model name (default: all models). "
            "e.g. 'llama3' to include only Llama3 family, "
            "'llama|qwen' to include both Llama and Qwen families, "
            "'^(?!.*deepseek)' to exclude DeepSeek family"
        ),
    )
    parser.add_argument(
        "-tp",
        "--tensor-parallelism",
        type=positive_int,
        nargs="+",
        choices=get_args(TpDegree),
        default=get_args(TpDegree),
        help="tensor parallelism degrees (default: all)",
    )
    # Batch size arguments:
    parser.add_argument(
        "-bs",
        "--batch-start",
        type=positive_int,
        default=DEFAULT_BATCH_START,
        help=f"initial batch size (inclusive, default: {DEFAULT_BATCH_START})",
    )
    parser.add_argument(
        "-bi",
        "--batch-inc",
        type=positive_int,
        default=DEFAULT_BATCH_INC,
        help=f"batch size increment (default: {DEFAULT_BATCH_INC})",
    )
    parser.add_argument(
        "-be",
        "--batch-end",
        type=positive_int,
        default=DEFAULT_BATCH_END,
        help=f"final batch size (inclusive, default: {DEFAULT_BATCH_END})",
    )
    # Sequence length arguments:
    parser.add_argument(
        "-ss",
        "--seq-start",
        type=positive_int,
        default=DEFAULT_SEQ_START,
        help=f"initial sequence length (inclusive, default: {DEFAULT_SEQ_START})",
    )
    parser.add_argument(
        "-si",
        "--seq-inc",
        type=positive_int,
        default=DEFAULT_SEQ_INC,
        help=f"sequence length increment (default: {DEFAULT_SEQ_INC})",
    )
    parser.add_argument(
        "-se",
        "--seq-end",
        type=positive_int,
        default=DEFAULT_SEQ_END,
        help=f"final sequence length (inclusive, default: {DEFAULT_SEQ_END})",
    )
    parser.add_argument(
        "-M",
        "--metric",
        type=str.lower,
        choices=sorted(METRICS.keys()),
        default="time",
        help="metric to benchmark (default: time)",
    )
    default_output: str = os.path.splitext(os.path.basename(__file__))[0] + ".csv"
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help=f"output CSV file with benchmark results (default: {default_output})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="list available models and exit",
    )
    parser.add_argument(
        "-L",
        "--log-level",
        type=str.lower,
        choices=["critical", "error", "warning", "info", "debug", "off"],
        default="info",
        help="log level to enable (default: info)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Validate range constraints:
    if args.batch_end < args.batch_start:
        parser.error("--batch-end must be greater than or equal to --batch-start")
    if args.seq_end < args.seq_start:
        parser.error("--seq-end must be greater than or equal to --seq-start")

    # Deduplicate and sort multi-value arguments:
    args.kernel = sorted(set(args.kernel))
    args.layout = sorted(set(args.layout))
    args.tensor_parallelism = sorted(set(args.tensor_parallelism))

    # Convert metric string to metric object:
    args.metric = METRICS[args.metric]

    # Convert string log level to numeric log level:
    args.log_level = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "off": logging.CRITICAL + 1000,
    }[args.log_level]

    return args


def get_bench_args_from_cli(args: argparse.Namespace) -> list[BenchArgs]:
    logging.debug("Requested kernels: %s", args.kernel)

    logging.debug("Requested layouts: %s", args.layout)

    logging.debug("Requested model filter: %s", args.model)
    filtered_models: list[Model] = get_models(args.model)
    model_names: list[str] = [model.name for model in filtered_models]
    logging.debug("Resolved model names: %s", model_names)

    logging.debug("Requested tensor parallelism: %s", args.tensor_parallelism)

    logging.debug(
        "Requested batch range: start=%d inc=%d end=%d",
        args.batch_start,
        args.batch_inc,
        args.batch_end,
    )

    logging.debug(
        "Requested seq. length range: start=%d inc=%d end=%d",
        args.seq_start,
        args.seq_inc,
        args.seq_end,
    )

    metric: Metric = args.metric
    logging.debug("Performance metric is %s in %s.", metric.name, metric.user_unit)

    logging.debug("Output data will be saved to [%s] file.", args.output)

    return get_bench_args(
        kernels=args.kernel,
        layouts=args.layout,
        tp_models=get_tp_models(models=filtered_models, tps=args.tensor_parallelism),
        batch_range=Range(
            start=args.batch_start,
            inc=args.batch_inc,
            end=args.batch_end,
        ),
        seq_range=Range(
            start=args.seq_start,
            inc=args.seq_inc,
            end=args.seq_end,
        ),
    )


def main() -> None:
    start_timestamp: float = time.perf_counter()

    args: argparse.Namespace = parse_args()

    disable_logs("matplotlib")
    logging.basicConfig(format="%(levelname)s|%(message)s", level=args.log_level)

    if args.list_models:
        list_models()
        return

    logging.info("Benchmarking attention configurations...")

    metric: Metric = args.metric
    if metric != METRICS["time"] and any(
        model.use_mla for model in get_models(args.model)
    ):
        metric = METRICS["time"]
        logging.warning(
            "One or more benchmarks are backed by MLA. MLA benchmark doesn't support throughput or bandwidth metrics, switching to time metric."
        )

    bench_args: list[BenchArgs] = get_bench_args_from_cli(args)
    num_bench_args: int = len(bench_args)
    logging.info("Number of benchmark configurations: %d", num_bench_args)
    if num_bench_args == 0:
        logging.warning(
            "There's no valid benchmark configuration for the given input combination."
        )
        return

    global_stats = GlobalStats()

    with open(args.output, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(BenchArgs.csv_header(metric))

        for ba_i, ba in enumerate(bench_args, start=1):
            perf: Optional[float] = run_bench(ba, metric)
            m: Model = ba.tp_model.model
            if perf is None:
                global_stats.report_failure(ba.kernel, m.name)
            else:
                global_stats.report_success(ba.kernel, m.name)
                logging.debug(
                    "%04d performance of %s is %.3f %s.",
                    ba_i,
                    ba.to_log_str(),
                    perf,
                    metric.user_unit,
                )
            writer.writerow(ba.csv_data(perf))

    global_stats.log_stats()

    end_timestamp: float = time.perf_counter()
    elapsed_time_s: float = end_timestamp - start_timestamp
    elapsed_time_hms: str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
    logging.info("Finished, execution took %s hh:mm:ss.", elapsed_time_hms)


if __name__ == "__main__":
    main()
