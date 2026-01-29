from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
import io
import logging
import shlex
from typing import Literal


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


@dataclass(kw_only=True)
class TpModel:
    model: Model
    tp: int = 1

    def __post_init__(self) -> None:
        assert self.tp > 0, "Tensor parallelism must be positive."
        assert (
            self.model.hq % self.tp == 0
        ), "Number of query heads must be divisible by tensor parallelism."

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


def run_bench_mha(args: BenchArgs) -> tuple[str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        bench_mha_main(shlex.split(args.to_cli_str()))
    return stdout.getvalue(), stderr.getvalue()


def main() -> None:
    m = Model(name="Llama3 405B", hq=128, hkv=8, dqk=128, dv=128)
    tpm = TpModel(model=m, tp=8)
    ba = BenchArgs(kernel="fwd", layout="thd", tp_model=tpm, b=1, s=1024)
    stdout, stderr = run_bench_mha(ba)
    if stderr:
        print(f"Errors / warnings:\n{stderr}")
    print(f"Output:\n{stdout}")


if __name__ == "__main__":
    main()
