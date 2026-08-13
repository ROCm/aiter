# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Balanced GDN prefill benchmark: production/native WS versus AITER Triton.

Examples:
    python op_tests/triton_tests/bench_opus_gdn_prefill.py --suite quick
    python op_tests/triton_tests/bench_opus_gdn_prefill.py --suite varlen
    python op_tests/triton_tests/bench_opus_gdn_prefill.py \
        --suite varlen --metadata-mode inference --wall

Input generation, correctness, warmup, and JIT compilation are outside the
timed region. Every provider gets a distinct preallocated output buffer. GPU
event samples are launched in balanced rotating/reversed order to limit clock
and temperature bias. ``--wall`` additionally reports synchronized caller
latency after the metadata cache has been populated. Cold metadata construction
is intentionally outside the timed region.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aiter.ops.gdn_prefill import gdn_prefill, select_gdn_prefill_path
from aiter.ops.opus_gdn_wu_prefill import (
    OPUS_GDN_K2_SPLIT,
    opus_gdn_wu_prefill_fwd,
)
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

_D = 128


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    lengths: tuple[int, ...]
    heads: int
    state_io: bool = False
    packed: bool = False

    @property
    def tokens(self) -> int:
        return sum(self.lengths) if self.packed else self.batch * self.lengths[0]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        if self.packed:
            return (1, sum(self.lengths), self.heads, _D)
        return (self.batch, self.lengths[0], self.heads, _D)

    @property
    def state_batch(self) -> int:
        return len(self.lengths) if self.packed else self.batch


CASES = (
    Case("dense-b1-t128-h8", 1, (128,), 8),
    Case("dense-b4-t2048-h16", 4, (2048,), 16),
    Case("dense-b1-t8192-h32-state", 1, (8192,), 32, state_io=True),
    Case("varlen-tail-1728-h16", 1, (63, 64, 65, 511, 1025), 16, packed=True),
    Case("varlen-ragged-1200-h4", 1, (15, 85, 200, 900), 4, packed=True),
    Case("varlen-aligned-8192-h16", 1, (1024,) * 8, 16, packed=True),
    Case(
        "varlen-boundary-5969-h8",
        1,
        (17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025),
        8,
        packed=True,
    ),
    Case(
        "varlen-skew-8192-h4",
        1,
        (1,) * 15 + (8177,),
        4,
        packed=True,
    ),
    Case(
        "varlen-tail-7809-h32-state",
        1,
        (127, 511, 1025, 2049, 4097),
        32,
        state_io=True,
        packed=True,
    ),
)


def _cu_from_lengths(lengths: tuple[int, ...], *, inference: bool) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    if inference:
        with torch.inference_mode():
            return torch.tensor(values, dtype=torch.int32, device="cuda")
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _make_inputs(
    case: Case,
    scalar_dtype: str,
    metadata_mode: str,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260804 + case.tokens + case.heads)
    q = torch.randn(case.shape, dtype=torch.bfloat16, device="cuda") / math.sqrt(_D)
    k = torch.randn(case.shape, dtype=torch.bfloat16, device="cuda") / math.sqrt(_D)
    v = torch.randn(case.shape, dtype=torch.bfloat16, device="cuda") * 0.1
    scalar_shape = case.shape[:-1]
    g = F.logsigmoid(torch.randn(scalar_shape, dtype=torch.float32, device="cuda"))
    beta = torch.sigmoid(torch.randn_like(g))
    if scalar_dtype == "mixed":
        beta = beta.to(torch.bfloat16)
    elif scalar_dtype == "bf16":
        g = g.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
    cu_seqlens = (
        _cu_from_lengths(
            case.lengths,
            inference=metadata_mode == "inference",
        )
        if case.packed
        else None
    )
    initial_state = (
        torch.randn(
            case.state_batch,
            case.heads,
            _D,
            _D,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.01
        if case.state_io
        else None
    )
    return q, k, v, g, beta, cu_seqlens, initial_state


def _balanced_orders(names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    rotations = tuple(names[i:] + names[:i] for i in range(len(names)))
    reverse = names[::-1]
    reverse_rotations = tuple(reverse[i:] + reverse[:i] for i in range(len(reverse)))
    return tuple(dict.fromkeys(rotations + reverse_rotations))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def _gpu_timing(
    providers: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]],
    warmup: int,
    repeat: int,
) -> dict[str, list[float]]:
    names = tuple(providers)
    orders = _balanced_orders(names)
    pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for name in names:
        for _ in range(warmup):
            providers[name]()
    torch.cuda.synchronize()
    for iteration in range(repeat):
        for name in orders[iteration % len(orders)]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            providers[name]()
            end.record()
            pending.append((name, start, end))
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for name, start, end in pending:
        samples[name].append(start.elapsed_time(end) * 1000.0)
    return samples


def _wall_timing(
    providers: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]],
    repeat: int,
) -> dict[str, list[float]]:
    names = tuple(providers)
    orders = _balanced_orders(names)
    samples = {name: [] for name in names}
    for iteration in range(repeat):
        for name in orders[iteration % len(orders)]:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            providers[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter_ns() - start) / 1000.0)
    return samples


def _make_providers(
    case: Case,
    inputs: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
) -> tuple[
    dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]],
    str,
]:
    q, k, v, g, beta, cu_seqlens, initial_state = inputs
    outputs = {name: torch.empty_like(v) for name in names}
    common = {
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "output_final_state": case.state_io,
        "cu_seqlens": cu_seqlens,
    }
    route = select_gdn_prefill_path(q, k, v, **common)
    providers: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]] = {}
    if "auto" in names:
        providers["auto"] = lambda: gdn_prefill(
            q,
            k,
            v,
            o=outputs["auto"],
            path="auto",
            **common,
        )
    if "native-ws" in names:
        providers["native-ws"] = lambda: opus_gdn_wu_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            out=outputs["native-ws"],
            initial_state=initial_state,
            output_final_state=case.state_io,
            k2_mode=OPUS_GDN_K2_SPLIT,
            use_env_overrides=False,
            cu_seqlens=cu_seqlens,
        )
    if "triton" in names:
        providers["triton"] = lambda: chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            o=outputs["triton"],
            **common,
        )
    return providers, route


def _check_correctness(
    providers: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]],
) -> tuple[float, float] | None:
    if "triton" not in providers:
        return None
    expected_o, expected_state = providers["triton"]()
    torch.cuda.synchronize()
    max_output = 0.0
    max_state = 0.0
    for name, provider in providers.items():
        if name == "triton":
            continue
        actual_o, actual_state = provider()
        torch.testing.assert_close(actual_o, expected_o, rtol=1e-2, atol=1e-2)
        max_output = max(
            max_output,
            float((actual_o.float() - expected_o.float()).abs().max()),
        )
        if expected_state is not None:
            assert actual_state is not None
            torch.testing.assert_close(
                actual_state,
                expected_state,
                rtol=1e-2,
                atol=2e-3,
            )
            max_state = max(
                max_state,
                float((actual_state - expected_state).abs().max()),
            )
    torch.cuda.synchronize()
    return max_output, max_state


def _selected_cases(suite: str, filters: list[str]) -> tuple[Case, ...]:
    selected = CASES
    if suite == "quick":
        selected = (CASES[0], CASES[3], CASES[5], CASES[7])
    elif suite == "dense":
        selected = tuple(case for case in CASES if not case.packed)
    elif suite == "varlen":
        selected = tuple(case for case in CASES if case.packed)
    if filters:
        selected = tuple(
            case for case in selected if any(text in case.name for text in filters)
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("quick", "full", "dense", "varlen"),
        default="quick",
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("auto", "native-ws", "triton"),
        default=("auto", "native-ws", "triton"),
    )
    parser.add_argument(
        "--scalar-dtype",
        choices=("fp32", "mixed", "bf16"),
        default="mixed",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=("versioned", "inference"),
        default="versioned",
    )
    parser.add_argument("--wall", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("a ROCm GPU is required")
    names = tuple(dict.fromkeys(args.providers))
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gfx = properties.gcnArchName.split(":", 1)[0]
    if "native-ws" in names and gfx not in ("gfx942", "gfx950"):
        raise RuntimeError("native-ws requires ROCm gfx942 or gfx950")
    cases = _selected_cases(args.suite, args.case)
    if not cases:
        raise ValueError("no benchmark cases matched")
    print(
        f"GPU={properties.name} gfx={gfx} CU={properties.multi_processor_count} "
        f"scalar={args.scalar_dtype} metadata={args.metadata_mode}"
    )
    print(
        "| case | auto route | provider | median us | p20 us | p80 us | "
        "Mtoken/s | vs Triton | wall median us |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for case in cases:
        inputs = _make_inputs(case, args.scalar_dtype, args.metadata_mode)
        with torch.inference_mode():
            providers, route = _make_providers(case, inputs, names)
            correctness = _check_correctness(providers)
            gpu = _gpu_timing(providers, args.warmup, args.repeat)
            wall = _wall_timing(providers, args.repeat) if args.wall else None
            triton_median = (
                statistics.median(gpu["triton"]) if "triton" in gpu else None
            )
            for name in names:
                values = gpu[name]
                median = statistics.median(values)
                speedup = (
                    f"{triton_median / median:.3f}x"
                    if triton_median is not None
                    else "N/A"
                )
                wall_median = f"{statistics.median(wall[name]):.3f}" if wall else "N/A"
                print(
                    f"| {case.name} | {route} | {name} | {median:.3f} | "
                    f"{_percentile(values, 0.2):.3f} | "
                    f"{_percentile(values, 0.8):.3f} | "
                    f"{case.tokens / median:.3f} | {speedup} | "
                    f"{wall_median} |"
                )
            if correctness is None:
                print(
                    f"correctness {case.name}: skipped (Triton not selected)",
                    flush=True,
                )
            else:
                max_output, max_state = correctness
                print(
                    f"correctness {case.name}: max_output={max_output:.9g} "
                    f"max_state={max_state:.9g}",
                    flush=True,
                )
        del providers, inputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
