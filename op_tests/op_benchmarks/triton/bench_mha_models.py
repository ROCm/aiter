import copy
import io
import logging
import shlex
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Literal, Optional, Self, get_args


def disable_aiter_logs() -> None:
    logging.getLogger("aiter").disabled = True


disable_aiter_logs()
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


@dataclass(kw_only=True)
class TpModel:
    model: Model
    tp: int = 1

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
    return get_bench_result(args, out.getvalue(), err.getvalue())


def get_models() -> Iterable[Model]:
    return (
        Model.new("Llama3 405B").h_q_vk(128, 8).d(128).build(),
        Model.new("Llama3 70B").h_q_vk(64, 8).d(128).build(),
        Model.new("Llama3 8B").h_q_vk(32, 8).d(128).build(),
        Model.new("Llama4 Maverick (Text)").h_q_vk(40, 8).d(128).build(),
        Model.new("Llama4 Maverick (Vision)").h(16).d(88).build(),
        Model.new("Qwen-235B-A22B").h_q_vk(64, 4).d(128).build(),
        Model.new("GPT-OSS 120B").h_q_vk(64, 8).d(64).build(),
        Model.new("DeepSeek R1").h(128).d_qk_v(192, 128).build(),
    )


def get_tp_models(
    models: Iterable[Model] = get_models(), tps: Iterable[int] = (1, 2, 4, 8)
) -> Iterable[TpModel]:
    return (TpModel(model=model, tp=tp) for model, tp in product(models, tps))


def get_bench_args(
    kernels: Iterable[Kernel] = get_args(Kernel),
    layouts: Iterable[Layout] = get_args(Layout),
    tp_models: Iterable[TpModel] = get_tp_models(),
) -> Iterable[BenchArgs]:
    return (
        BenchArgs(kernel=kernel, layout=layout, tp_model=tp_model, b=1, s=1024)
        for kernel, layout, tp_model in product(kernels, layouts, tp_models)
    )


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Benchmarking attention configurations...")
    for ba in get_bench_args():
        ms = run_bench_mha(ba)
        if ms is not None:
            logging.info("%s => error!", ba)
        else:
            logging.info("%s => %.3f ms", ba, ms)
    logging.info("DONE.")


if __name__ == "__main__":
    main()
