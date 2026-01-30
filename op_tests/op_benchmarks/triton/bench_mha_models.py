import argparse
import copy
import io
import logging
import shlex
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Literal, Optional, Self, get_args

import matplotlib.pyplot as plt


def disable_logs(logger: str) -> None:
    logging.getLogger(logger).disabled = True
    logging.getLogger(logger).setLevel(logging.CRITICAL + 1)


disable_logs("aiter")
from bench_mha import main as bench_mha_main  # noqa: E402

Flavor = Literal["mha", "gqa", "mla"]


@dataclass(kw_only=True)
class Model:
    name: str
    hq: int
    hkv: int
    dqk: int
    dv: int
    flavor: Flavor = field(init=False)

    def __post_init__(self) -> None:
        assert self.name, "Model name must be non-empty."
        assert self.hq > 0, "Number of query heads must be positive."
        assert self.hkv > 0, "Number of key and value heads must be positive."
        assert self.dqk > 0, "Dimension of query and key heads must be positive."
        assert self.dv > 0, "Dimension of value heads must be positive."

        flavor: Flavor
        if self.hq == self.hkv and self.dqk == self.dv:
            flavor = "mha"
        elif self.hq == self.hkv and self.dqk > self.dv:
            flavor = "mla"
        elif self.hq > self.hkv and self.hq % self.hkv == 0 and self.dqk == self.dv:
            flavor = "gqa"
        else:
            assert False, "Unable to deduce attention flavor from heads configuration."
        self.flavor = flavor

    @classmethod
    def new(cls, name: str) -> "ModelBuilder":
        return ModelBuilder(name)


class ModelBuilder:
    name: str
    hq: int
    hkv: int
    dqk: int
    dv: int

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

    def build(self) -> Model:
        return Model(name=self.name, hq=self.hq, hkv=self.hkv, dqk=self.dqk, dv=self.dv)


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

        self.model = copy.copy(self.model)
        self.model.hq = self.model.hq // self.tp
        self.model.hkv = max(self.model.hkv // self.tp, 1)


# There are two backward implementations:
# * "one kernel", the default one, refered as "bwdo"
# * "fused", the legacy one, refered as "bwdf"
Kernel = Literal["fwd", "bwdo", "bwdf"]


Layout = Literal["bshd", "thd"]


@dataclass(kw_only=True)
class BenchArgs:
    kernel: Kernel
    layout: Layout
    tp_model: TpModel
    b: int
    s: int

    def __post_init__(self) -> None:
        assert self.b > 0, "Batch size must be positive."
        assert self.s > 0, "Sequence length must be positive."

    def to_cli_str(self) -> str:
        s: str = str(self.s)
        m: Model = self.tp_model.model
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
            "-d": str(m.dqk),
            "-dv": str(m.dv),
            "-metric": "time",
        }
        args_list: list[str] = [kv for k, v in args_dict.items() for kv in (k, v)]
        if self.kernel == "bwdf":
            args_list.append("-fused_bwd")
        args_str: str = " ".join(args_list)
        return args_str


def get_bench_result(args: BenchArgs, out: str, err: str) -> Optional[float]:
    # Check empty stderr:
    if err:
        logging.debug("Standard error stream isn't empty: [%s]", err)
        return None
    # Split stdout:
    out_lines: list[list[str]] = [
        out_line.split() for out_line in out.strip().split(sep="\n")
    ]
    # Check number of lines in stdout:
    if len(out_lines) != 3:
        logging.debug("Standard out stream doesn't have 3 lines: [%s]", out)
        return None
    l0: list[str]
    l1: list[str]
    l2: list[str]
    l0, l1, l2 = out_lines
    # Check stdout line #1 (benchmark name):
    if l0 != ["bench_mha:"]:
        logging.debug("Benchmark name doesn't match: %s", l0)
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
        f"{kernel_header}(ms)",
        "(ms)",
    ]:
        logging.debug("Table header doesn't match: %s", l1)
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
            logging.debug("Table data doesn't match: %s", l2)
            return None
        return float(l2[6])
    except ValueError:
        logging.exception("Unexpected numeric conversion error.")
        return None


def run_bench_mha(args: BenchArgs) -> Optional[float]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        bench_mha_main(shlex.split(args.to_cli_str()))
    # Close matplotlib figures to silence errors and avoid memory leaks.
    plt.close("all")
    return get_bench_result(args, out.getvalue(), err.getvalue())


def get_models(model_filter: Optional[str] = None) -> Iterable[Model]:
    all_models = (
        Model.new("Llama3 405B").h_q_vk(128, 8).d(128).build(),
        Model.new("Llama3 70B").h_q_vk(64, 8).d(128).build(),
        Model.new("Llama3 8B").h_q_vk(32, 8).d(128).build(),
        Model.new("Llama4 Maverick (Text)").h_q_vk(40, 8).d(128).build(),
        Model.new("Llama4 Maverick (Vision)").h(16).d(88).build(),
        Model.new("Qwen-235B-A22B").h_q_vk(64, 4).d(128).build(),
        Model.new("GPT-OSS 120B").h_q_vk(64, 8).d(64).build(),
        Model.new("DeepSeek R1 (Prefill)").h(128).d(56).build(),
        Model.new("DeepSeek R1 (Decode)").h(128).d_qk_v(192, 128).build(),
    )
    if model_filter is not None:
        model_filter = model_filter.strip().lower()
        if not model_filter:  # Empty string after stripping
            logging.debug("Empty model name filter, returning all models.")
            return all_models
        filtered_models = tuple(
            model for model in all_models if model_filter in model.name.lower()
        )
        logging.debug("Number of filtered models: %d", len(filtered_models))
        if not filtered_models:
            logging.warning("There are no models after filtering by model name.")
        return filtered_models
    return all_models  # model_filter is None, return all


def get_tp_models(
    models: Iterable[Model] = get_models(),
    tps: Iterable[TpDegree] = get_args(TpDegree),
) -> Iterable[TpModel]:
    return tuple(TpModel(model=model, tp=tp) for model, tp in product(models, tps))


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
    tp_models: Iterable[TpModel] = get_tp_models(),
    batch_range: Range = Range(start=1, inc=1, end=4),
    seq_range: Range = Range(start=1024, inc=1024, end=5120),
) -> Iterable[BenchArgs]:
    return tuple(
        BenchArgs(kernel=kernel, layout=layout, tp_model=tp_model, b=b, s=s)
        for kernel, layout, tp_model, b, s in product(
            kernels,
            layouts,
            tp_models,
            batch_range.to_range(),
            seq_range.to_range(),
        )
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
        nargs="+",
        choices=get_args(Kernel),
        default=get_args(Kernel),
        help="attention kernels (default: all)",
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="+",
        choices=get_args(Layout),
        default=get_args(Layout),
        help="memory layouts (default: all)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="model name filter (case-insensitive substring match, default: all models)",
    )
    parser.add_argument(
        "-t",
        "--tensor-parallelism",
        nargs="+",
        type=positive_int,
        choices=get_args(TpDegree),
        default=get_args(TpDegree),
        help="tensor parallelism degrees (default: all)",
    )
    # Batch size arguments:
    parser.add_argument(
        "-bs",
        "--batch-start",
        type=positive_int,
        default=1,
        help="initial batch size (inclusive, default: 1)",
    )
    parser.add_argument(
        "-bi",
        "--batch-inc",
        type=positive_int,
        default=1,
        help="batch size increment (default: 1)",
    )
    parser.add_argument(
        "-be",
        "--batch-end",
        type=positive_int,
        default=4,
        help="final batch size (inclusive, default: 4)",
    )
    # Sequence length arguments:
    parser.add_argument(
        "-ss",
        "--seq-start",
        type=positive_int,
        default=1024,
        help="initial sequence length (inclusive, default: 1024)",
    )
    parser.add_argument(
        "-si",
        "--seq-inc",
        type=positive_int,
        default=1024,
        help="sequence length increment (default: 1024)",
    )
    parser.add_argument(
        "-se",
        "--seq-end",
        type=positive_int,
        default=5120,
        help="final sequence length (inclusive, default: 5120)",
    )

    args = parser.parse_args()

    # Validate range constraints:
    if args.batch_end < args.batch_start:
        parser.error("--batch-end must be greater than or equal to --batch-start")
    if args.seq_end < args.seq_start:
        parser.error("--seq-end must be greater than or equal to --seq-start")

    # Deduplicate and sort multi-value arguments:
    args.kernel = sorted(set(args.kernel))
    args.layout = sorted(set(args.layout))
    args.tensor_parallelism = sorted(set(args.tensor_parallelism))

    return args


def main() -> None:
    args = parse_args()

    disable_logs("matplotlib")
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Benchmarking attention configurations...")

    bench_args = get_bench_args(
        kernels=args.kernel,
        layouts=args.layout,
        tp_models=get_tp_models(
            models=get_models(args.model),
            tps=args.tensor_parallelism,
        ),
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

    for ba in bench_args:
        ms = run_bench_mha(ba)
        if ms is None:
            logging.info("%s => error!", ba)
        else:
            logging.info("%s => %.3f ms", ba, ms)

    logging.info("DONE.")


if __name__ == "__main__":
    main()
