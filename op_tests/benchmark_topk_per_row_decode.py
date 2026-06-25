import argparse
import csv
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


TOPK_CO_PATH = Path("aiter/hsa/gfx942/topk_per_row_decode/asm_top_k_per_row_decode.co")


@dataclass(frozen=True)
class Shape:
    batch_size: int
    num_rows: int
    max_model_len: int
    next_n: int
    k: int
    stride_mode: str


@dataclass
class Kernel:
    name: str
    available: bool
    note: str
    runner: Callable[..., None] | None


def parse_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def csv_emit(writer: csv.writer, row: list[object]) -> None:
    writer.writerow(row)
    sys.stdout.flush()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_path_if_present(path: str | None) -> None:
    if not path:
        return
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def add_repo_to_python_path() -> None:
    add_path_if_present(str(repo_root()))


def arch_name() -> str:
    if not torch.cuda.is_available():
        return "unavailable"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gcn_arch = getattr(props, "gcnArchName", "")
    if gcn_arch:
        return gcn_arch.split(":", maxsplit=1)[0]
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx()
    except Exception:
        return props.name


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype for decode topk kernels: {name}")


def make_row_ends(
    batch_size: int,
    num_rows: int,
    max_model_len: int,
    next_n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_lens = torch.full((batch_size,), max_model_len, dtype=torch.int32, device=device)
    row_ids = torch.arange(num_rows, dtype=torch.int64, device=device)
    batch_ids = row_ids // next_n
    next_offsets = row_ids % next_n
    row_ends = seq_lens[batch_ids] - next_n + next_offsets + 1
    return seq_lens, row_ends.to(torch.int32)


def make_values(
    shape: tuple[int, int],
    dtype: torch.dtype,
    distribution: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if distribution == "random":
        return torch.randn(shape, dtype=dtype, device=device)
    if distribution in {"10LSBits", "mixed"}:
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        random_bottom_bits = torch.randint(
            0,
            2**10,
            shape,
            dtype=torch.int32,
            device=device,
        )
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)
        if distribution == "mixed":
            mask = torch.randint(0, 2, (shape[0], 1), device=device).bool()
            logits = torch.where(mask, logits, torch.randn(shape, dtype=dtype, device=device))
        return logits
    if distribution == "ties":
        return torch.randint(-16, 16, shape, dtype=torch.int32, device=device).to(dtype)
    raise ValueError(f"unknown distribution: {distribution}")


def make_logits(
    num_rows: int,
    max_model_len: int,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    distribution: str,
    stride_mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if stride_mode == "unit":
        logits = make_values(
            (num_rows, max_model_len),
            dtype,
            distribution,
            seed,
            device,
        )
    elif stride_mode == "nonunit":
        base = make_values(
            (num_rows, max_model_len * 2),
            dtype,
            distribution,
            seed,
            device,
        )
        logits = base[:, ::2]
    else:
        raise ValueError(f"unknown stride mode: {stride_mode}")

    for row, end in enumerate(row_ends.tolist()):
        if end < max_model_len:
            logits[row, end:] = float("-inf")
    return logits


def torch_reference(logits: torch.Tensor, row_ends: torch.Tensor, k: int) -> torch.Tensor:
    ref_k = min(k, logits.shape[1])
    indices = torch.topk(logits, ref_k, dim=-1).indices.to(torch.int32)
    if ref_k == k:
        return indices
    padded = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device=logits.device)
    padded[:, :ref_k] = indices
    return padded


def compare_indices(
    logits: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    tolerance: float,
) -> tuple[bool, str]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    row_ends_cpu = row_ends.detach().cpu()

    for row in range(actual_cpu.shape[0]):
        row_end = int(row_ends_cpu[row].item())
        valid_count = min(k, row_end)
        actual_row = actual_cpu[row, :valid_count]
        expected_row = expected_cpu[row, :valid_count]

        invalid = (actual_row < 0) | (actual_row >= row_end)
        if invalid.any():
            bad = actual_row[invalid][:4].tolist()
            return False, f"invalid indices in row {row}: {bad}"

        actual_list = actual_row.tolist()
        expected_list = expected_row.tolist()
        actual_set = set(actual_list)
        expected_set = set(expected_list)

        if len(actual_set) != len(actual_list):
            return False, f"duplicate indices in row {row}"
        if actual_set == expected_set:
            continue

        actual_only = list(actual_set - expected_set)
        expected_only = list(expected_set - actual_set)
        if len(actual_only) != len(expected_only):
            return False, f"different topk set sizes in row {row}"

        row_logits = logits[row].detach().cpu()
        actual_values = torch.tensor([row_logits[i].item() for i in actual_only])
        expected_values = torch.tensor([row_logits[i].item() for i in expected_only])
        if not torch.allclose(
            actual_values,
            expected_values,
            rtol=tolerance,
            atol=tolerance,
        ):
            return False, f"value mismatch in row {row}"
    return True, "ok"


def time_kernel(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    enabled: bool,
) -> str:
    if not enabled:
        fn()
        torch.cuda.synchronize()
        return "na"

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return f"{start.elapsed_time(end) * 1000.0 / iters:.3f}"


def load_aiter_module() -> tuple[object | None, str]:
    try:
        import aiter

        return aiter, ""
    except Exception as exc:
        return None, f"aiter import failed: {type(exc).__name__}: {exc}"


def load_vllm_kernel(vllm_path: str | None) -> Kernel:
    add_path_if_present(vllm_path)
    try:
        custom_ops = importlib.import_module("vllm._custom_ops")
        runner = custom_ops.top_k_per_row_decode
        return Kernel("vllm_rocm", True, "vllm._custom_ops", runner)
    except Exception as custom_exc:
        try:
            runner = torch.ops._C.top_k_per_row_decode
            return Kernel("vllm_rocm", True, "torch.ops._C", runner)
        except Exception as ops_exc:
            return Kernel(
                "vllm_rocm",
                False,
                (
                    "unavailable: vllm._custom_ops "
                    f"{type(custom_exc).__name__}; torch.ops._C {type(ops_exc).__name__}"
                ),
                None,
            )


def load_flydsl_kernel(symbol: str | None, flydsl_path: str | None) -> Kernel:
    if not symbol:
        return Kernel("flydsl", False, "not configured; pass --flydsl-symbol module:function", None)
    add_path_if_present(flydsl_path)
    try:
        module_name, func_name = symbol.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        runner = getattr(module, func_name)
        return Kernel("flydsl", True, symbol, runner)
    except Exception as exc:
        return Kernel(
            "flydsl",
            False,
            f"unavailable: {type(exc).__name__}: {exc}",
            None,
        )


def selected(name: str, kernels: set[str]) -> bool:
    return "all" in kernels or name in kernels


def build_kernels(args: argparse.Namespace, arch: str) -> list[Kernel]:
    kernels: list[Kernel] = []
    selected_kernels = set(args.kernels)

    if selected("torch", selected_kernels):
        kernels.append(Kernel("torch_topk", True, "correctness reference only", None))

    if selected("vllm", selected_kernels):
        kernels.append(load_vllm_kernel(args.vllm_path))

    co_path = repo_root() / TOPK_CO_PATH
    needs_aiter = selected("aiter_hip", selected_kernels) or (
        selected("aiter_asm", selected_kernels) and arch == "gfx942" and co_path.exists()
    )
    aiter_mod, aiter_note = load_aiter_module() if needs_aiter else (None, "")
    if selected("aiter_hip", selected_kernels):
        if aiter_mod is None:
            kernels.append(Kernel("aiter_hip", False, aiter_note, None))
        else:
            kernels.append(Kernel("aiter_hip", True, "aiter.top_k_per_row_decode", aiter_mod.top_k_per_row_decode))

    if selected("aiter_asm", selected_kernels):
        if arch != "gfx942":
            kernels.append(Kernel("aiter_asm", False, "gfx942-only fast path", None))
        elif not co_path.exists():
            kernels.append(Kernel("aiter_asm", False, f"missing {TOPK_CO_PATH}", None))
        elif aiter_mod is None:
            kernels.append(Kernel("aiter_asm", False, aiter_note, None))
        else:
            kernels.append(
                Kernel(
                    "aiter_asm",
                    True,
                    "aiter.top_k_per_row_decode_fast",
                    aiter_mod.top_k_per_row_decode_fast,
                )
            )

    if selected("flydsl", selected_kernels):
        kernels.append(load_flydsl_kernel(args.flydsl_symbol, args.flydsl_path))

    return kernels


def run_kernel(
    kernel: Kernel,
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    k: int,
) -> None:
    assert kernel.runner is not None
    if kernel.name == "aiter_hip":
        kernel.runner(
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
        )
    elif kernel.name == "aiter_asm":
        kernel.runner(
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )
    else:
        kernel.runner(
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k,
        )


def shape_iter(args: argparse.Namespace) -> Iterable[Shape]:
    for next_n in args.next_n:
        if args.num_rows:
            row_specs = []
            for num_rows in args.num_rows:
                if num_rows % next_n != 0:
                    raise ValueError(
                        f"num_rows={num_rows} must be divisible by next_n={next_n}"
                    )
                row_specs.append((num_rows // next_n, num_rows))
        else:
            row_specs = [(batch, batch * next_n) for batch in args.batch_size]

        for batch_size, num_rows in row_specs:
            for max_model_len in args.max_model_len:
                if max_model_len < next_n:
                    raise ValueError(
                        f"max_model_len={max_model_len} must be >= next_n={next_n}"
                    )
                for k in args.k:
                    for stride_mode in args.stride_modes:
                        yield Shape(
                            batch_size=batch_size,
                            num_rows=num_rows,
                            max_model_len=max_model_len,
                            next_n=next_n,
                            k=k,
                            stride_mode=stride_mode,
                        )


def run_shape(
    writer: csv.writer,
    args: argparse.Namespace,
    arch: str,
    kernels: list[Kernel],
    shape: Shape,
) -> None:
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    seq_lens, row_ends = make_row_ends(
        shape.batch_size,
        shape.num_rows,
        shape.max_model_len,
        shape.next_n,
        device,
    )
    logits = make_logits(
        shape.num_rows,
        shape.max_model_len,
        row_ends,
        dtype,
        args.distribution,
        shape.stride_mode,
        args.seed,
        device,
    )
    expected = torch_reference(logits, row_ends, shape.k)

    for kernel in kernels:
        note = kernel.note
        if kernel.name == "torch_topk":
            csv_emit(
                writer,
                [
                    arch,
                    kernel.name,
                    shape.k,
                    shape.num_rows,
                    shape.max_model_len,
                    shape.next_n,
                    shape.stride_mode,
                    "na",
                    "reference",
                    note,
                ],
            )
            continue

        if kernel.name == "aiter_asm" and shape.k != 2048:
            csv_emit(
                writer,
                [
                    arch,
                    kernel.name,
                    shape.k,
                    shape.num_rows,
                    shape.max_model_len,
                    shape.next_n,
                    shape.stride_mode,
                    "na",
                    "unavailable",
                    "fast path only supports k=2048",
                ],
            )
            continue

        if not kernel.available:
            csv_emit(
                writer,
                [
                    arch,
                    kernel.name,
                    shape.k,
                    shape.num_rows,
                    shape.max_model_len,
                    shape.next_n,
                    shape.stride_mode,
                    "na",
                    "unavailable",
                    note,
                ],
            )
            continue

        indices = torch.empty((shape.num_rows, shape.k), dtype=torch.int32, device=device)

        def invoke() -> None:
            run_kernel(
                kernel,
                logits,
                shape.next_n,
                seq_lens,
                indices,
                shape.num_rows,
                shape.k,
            )

        try:
            latency = time_kernel(invoke, args.warmup, args.iters, not args.correctness_only)
            ok, compare_note = compare_indices(
                logits,
                indices,
                expected,
                row_ends,
                shape.k,
                args.tolerance,
            )
            correctness = "pass" if ok else "fail"
            if compare_note != "ok":
                note = f"{note}; {compare_note}"
        except Exception as exc:
            latency = "na"
            correctness = "error"
            note = f"{note}; {type(exc).__name__}: {exc}"

        csv_emit(
            writer,
            [
                arch,
                kernel.name,
                shape.k,
                shape.num_rows,
                shape.max_model_len,
                shape.next_n,
                shape.stride_mode,
                latency,
                correctness,
                note,
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMD-only decode top_k_per_row benchmark harness"
    )
    parser.add_argument("--k", type=int, nargs="+", default=[2048])
    parser.add_argument(
        "--max-model-len",
        "--context-len",
        dest="max_model_len",
        type=int,
        nargs="+",
        default=[4096],
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1])
    parser.add_argument("--num-rows", type=int, nargs="+", default=None)
    parser.add_argument("--next-n", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--stride-modes",
        nargs="+",
        choices=["unit", "nonunit"],
        default=["unit", "nonunit"],
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["all", "torch", "vllm", "aiter_hip", "aiter_asm", "flydsl"],
        default=["all"],
    )
    parser.add_argument(
        "--distribution",
        choices=["random", "10LSBits", "mixed", "ties"],
        default="random",
    )
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run each implementation once and skip latency timing.",
    )
    parser.add_argument(
        "--vllm-path",
        default=os.environ.get("VLLM_PATH", "/home/AMD/samremes/dev/vllm"),
        help="Optional vLLM checkout to add to PYTHONPATH for the ROCm baseline.",
    )
    parser.add_argument(
        "--flydsl-path",
        default=os.environ.get("FLYDSL_PATH", "/home/AMD/samremes/dev/FlyDSL"),
        help="Optional FlyDSL checkout to add to PYTHONPATH.",
    )
    parser.add_argument(
        "--flydsl-symbol",
        default=os.environ.get("FLYDSL_TOPK_PER_ROW_DECODE"),
        help="Optional module:function with the decode topk interface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.k = parse_int_list(args.k)
    args.max_model_len = parse_int_list(args.max_model_len)
    args.batch_size = parse_int_list(args.batch_size)
    args.next_n = parse_int_list(args.next_n)
    args.num_rows = None if args.num_rows is None else parse_int_list(args.num_rows)

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/HIP-visible AMD device is required.")

    add_repo_to_python_path()
    torch.set_default_device("cuda:0")
    arch = arch_name()
    kernels = build_kernels(args, arch)
    writer = csv.writer(sys.stdout)
    csv_emit(
        writer,
        [
            "arch",
            "kernel",
            "k",
            "num_rows",
            "max_model_len",
            "next_n",
            "stride_mode",
            "latency_us",
            "correctness",
            "notes",
        ],
    )
    for shape in shape_iter(args):
        run_shape(writer, args, arch, kernels, shape)


if __name__ == "__main__":
    main()
