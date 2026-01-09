# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Prebuild Gluon paged-attention decode AOT kernels (compile-only).

This script is intentionally lightweight and mirrors the style of
`aiter/aot/sampling.py`: generate a list of compile configs and compile them
in parallel.

Notes:
- We call `csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot.compile(...)` directly.
- We deduplicate by `func_name` (the final exported symbol name) to avoid
  compiling the same kernel multiple times.
- Parallelism is controlled by `MAX_JOBS` env var or `--max_jobs`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set


@dataclass(frozen=True)
class PaDecodeGluonCompileConfig:
    compute_type: str  # one of: "fp8", "bf16", "fp16"
    query_seq_len: int
    one_query_group_size: int
    head_size: int
    kv_block_size: int
    context_partition_size: int
    query_quant_mode: int  # -1 none, 0 per_tensor, 1 per_token
    kv_quant_mode: int  # -1 none, 0 per_tensor, 1 per_token
    fp8_max_value: float
    value_transposed: int
    is_causal: int
    use_sinks: int
    cdna_version: int
    func_name: str


def _quant_mode_to_int(enabled: bool, quant_mode: str) -> int:
    if not enabled:
        return -1
    if quant_mode == "per_tensor":
        return 0
    if quant_mode == "per_token":
        return 1
    raise ValueError(f"Unsupported quant_mode: {quant_mode}")


def _compute_func_name(
    *,
    md_name: str,
    compute_type_torch,
    query_seq_len: int,
    one_query_group_size: int,
    head_size: int,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
    use_sinks: int,
    cdna_version: int,
) -> str:
    import triton  # local import to keep top-level light
    from csrc.cpp_itfs.utils import get_default_func_name

    head_size_pow2 = triton.next_power_of_2(head_size)
    return get_default_func_name(
        md_name,
        (
            compute_type_torch,
            query_seq_len,
            one_query_group_size,
            head_size_pow2,
            kv_block_size,
            context_partition_size,
            query_quant_mode,
            kv_quant_mode,
            fp8_max_value,
            value_transposed,
            is_causal,
            use_sinks,
            cdna_version,
        ),
    )


def _iter_preset_params(preset: str) -> Iterable[dict]:
    """
    Yield parameter dicts describing what to compile.

    This replaces the old "global options + benchmark-style config generation"
    with explicit presets.
    """
    # Common defaults
    head_size = 128
    context_partition_size = 256
    use_sinks = 0
    value_transposed = 0

    if preset in ("performance", "perf"):
        # Old behavior (roughly): prebuild_normal_performance_cases_aot_so()
        # - context_length/batch_size were only used to produce a large variety
        #   of shapes for runtime tests; for AOT we only care about kernel
        #   compile parameters that affect codegen and symbol names.
        compute_types_quant = [("fp8", True, True), ("bf16", False, False)]
        quant_modes = ["per_tensor"]
        kv_block_sizes = [16, 64]
        query_seq_lens = [1, 2, 3, 4]
        head_configs = [(64, 4), (64, 8)]  # (q_heads, kv_heads)
    elif preset in ("accuracy", "acc"):
        # Old behavior (roughly): prebuild_normal_accuracy_cases_aot_so()
        compute_types_quant = [("fp8", True, True), ("bf16", False, False)]
        quant_modes = ["per_token", "per_tensor"]
        kv_block_sizes = [16, 64, 1024]
        query_seq_lens = [1, 2, 3, 4]
        head_configs = [(5, 1), (8, 1), (10, 1), (16, 1)]
    elif preset in ("all",):
        for p in ("accuracy", "performance"):
            yield from _iter_preset_params(p)
        return
    else:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from: accuracy, performance, all."
        )

    for compute_type, quant_q, quant_kv in compute_types_quant:
        for quant_mode in quant_modes:
            for kv_block_size in kv_block_sizes:
                for query_seq_len in query_seq_lens:
                    for q_heads, kv_heads in head_configs:
                        if q_heads % kv_heads != 0:
                            continue
                        one_query_group_size = q_heads // kv_heads
                        is_causal = int(query_seq_len > 1)
                        yield {
                            "compute_type": compute_type,
                            "quant_q": quant_q,
                            "quant_kv": quant_kv,
                            "quant_mode": quant_mode,
                            "query_seq_len": query_seq_len,
                            "one_query_group_size": one_query_group_size,
                            "head_size": head_size,
                            "kv_block_size": kv_block_size,
                            "context_partition_size": context_partition_size,
                            "value_transposed": value_transposed,
                            "is_causal": is_causal,
                            "use_sinks": use_sinks,
                        }


def generate_compile_configs(preset: str) -> List[PaDecodeGluonCompileConfig]:
    import torch  # local import
    import aiter  # local import
    from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import MD_NAME
    from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version

    cdna_version = int(get_cdna_version())

    def to_torch_dtype(name: str):
        if name == "fp8":
            return aiter.dtypes.fp8
        if name == "bf16":
            return torch.bfloat16
        if name == "fp16":
            return torch.float16
        raise ValueError(f"Unsupported compute_type: {name}")

    configs: List[PaDecodeGluonCompileConfig] = []
    seen: Set[str] = set()

    for params in _iter_preset_params(preset):
        compute_type_name = params["compute_type"]
        compute_type_torch = to_torch_dtype(compute_type_name)

        query_quant_mode = _quant_mode_to_int(params["quant_q"], params["quant_mode"])
        kv_quant_mode = _quant_mode_to_int(params["quant_kv"], params["quant_mode"])

        fp8_max_value = 1.0
        if kv_quant_mode >= 0:
            fp8_max_value = float(torch.finfo(aiter.dtypes.fp8).max)

        func_name = _compute_func_name(
            md_name=MD_NAME,
            compute_type_torch=compute_type_torch,
            query_seq_len=params["query_seq_len"],
            one_query_group_size=params["one_query_group_size"],
            head_size=params["head_size"],
            kv_block_size=params["kv_block_size"],
            context_partition_size=params["context_partition_size"],
            query_quant_mode=query_quant_mode,
            kv_quant_mode=kv_quant_mode,
            fp8_max_value=fp8_max_value,
            value_transposed=params["value_transposed"],
            is_causal=params["is_causal"],
            use_sinks=params["use_sinks"],
            cdna_version=cdna_version,
        )

        if func_name in seen:
            continue
        seen.add(func_name)

        configs.append(
            PaDecodeGluonCompileConfig(
                compute_type=compute_type_name,
                query_seq_len=params["query_seq_len"],
                one_query_group_size=params["one_query_group_size"],
                head_size=params["head_size"],
                kv_block_size=params["kv_block_size"],
                context_partition_size=params["context_partition_size"],
                query_quant_mode=query_quant_mode,
                kv_quant_mode=kv_quant_mode,
                fp8_max_value=fp8_max_value,
                value_transposed=params["value_transposed"],
                is_causal=params["is_causal"],
                use_sinks=params["use_sinks"],
                cdna_version=cdna_version,
                func_name=func_name,
            )
        )

    return configs


def _compile_one(config: PaDecodeGluonCompileConfig) -> None:
    import torch  # local import
    import aiter  # local import
    from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import compile as pa_compile

    if config.compute_type == "fp8":
        compute_type_torch = aiter.dtypes.fp8
    elif config.compute_type == "bf16":
        compute_type_torch = torch.bfloat16
    elif config.compute_type == "fp16":
        compute_type_torch = torch.float16
    else:
        raise ValueError(f"Unsupported compute_type: {config.compute_type}")

    pa_compile(
        compute_type=compute_type_torch,
        query_seq_len=config.query_seq_len,
        one_query_group_size=config.one_query_group_size,
        head_size=config.head_size,
        kv_block_size=config.kv_block_size,
        context_partition_size=config.context_partition_size,
        query_quant_mode=config.query_quant_mode,
        kv_quant_mode=config.kv_quant_mode,
        fp8_max_value=config.fp8_max_value,
        value_transposed=config.value_transposed,
        is_causal=config.is_causal,
        use_sinks=config.use_sinks,
        cdna_version=config.cdna_version,
        func_name=config.func_name,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prebuild pa_decode_gluon AOT kernels (compile-only)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=os.environ.get("AITER_PA_DECODE_GLUON_PREBUILD_PRESET", "performance"),
        choices=["accuracy", "performance", "all"],
        help="Which preset of kernels to compile",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=None,
        help="Max parallel compile jobs (overrides MAX_JOBS env)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print how many kernels would be compiled",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    configs = generate_compile_configs(args.preset)
    if args.dry_run:
        print(f"[dry_run] preset={args.preset} unique_kernels={len(configs)}")
        return

    max_jobs = args.max_jobs
    if max_jobs is None:
        max_jobs = int(os.environ.get("MAX_JOBS", os.cpu_count() or 16))
    max_jobs = max(1, max_jobs)

    print(f"preset={args.preset} unique_kernels={len(configs)} max_jobs={max_jobs}")

    # CUDA cannot be (re)initialized in a forked subprocess.
    # Use "spawn" for multiprocessing when running parallel compiles.
    if max_jobs == 1:
        for cfg in configs:
            _compile_one(cfg)
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_jobs,
            mp_context=ctx,
        ) as executor:
            list(executor.map(_compile_one, configs))

    print("All pa_decode_gluon AOT kernels built successfully!")


if __name__ == "__main__":
    main()
