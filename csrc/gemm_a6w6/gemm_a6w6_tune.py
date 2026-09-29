# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune registered gfx950 MXFP6 A6W6 assembly kernels by GEMM shape."""

import gc
import math
import os
import resource
import statistics
from collections.abc import Callable
from typing import Any, ClassVar, TypedDict

import pandas as pd
import torch

import aiter
from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_GEMM_A6W6, get_asm_dir
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

PACK_LAYOUT = "mxfp6_c0c1_256_padk2"
_MANIFEST_NAME = "f6gemm_bf16_per1x32Fp6.csv"
_MANIFEST_COLUMNS = frozenset(
    {
        "tile_M",
        "tile_N",
        "block_size",
        "splitK",
        "pack_layout",
        "swizzle_max_M",
        "swizzle_max_N",
        "swizzle_max_K",
        "knl_name",
        "co_name",
    }
)


class F6GemmCandidate(TypedDict):
    kernel_id: int
    kernel_name: str
    tile_m: int
    tile_n: int
    swizzle_max_m: int
    swizzle_max_n: int
    swizzle_max_k: int


def _disable_core_dumps() -> None:
    """Prevent a failed GPU tuning worker from filling shared node storage."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _manifest_path() -> str:
    return os.path.join(get_asm_dir(), "f6gemm", _MANIFEST_NAME)


def load_f6gemm_candidates() -> list[F6GemmCandidate]:
    manifest = _manifest_path()
    configs = pd.read_csv(manifest)
    missing = _MANIFEST_COLUMNS - set(configs.columns)
    if missing:
        raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")

    candidates: list[F6GemmCandidate] = []
    seen_names: set[str] = set()
    for kernel_id, row in configs.iterrows():
        kernel_name = str(row["knl_name"]).strip()
        co_name = str(row["co_name"]).strip()
        if not kernel_name or kernel_name in seen_names:
            raise ValueError(f"invalid or duplicate A6W6 kernel name: {kernel_name!r}")
        if not co_name:
            raise ValueError(f"{kernel_name} has an empty code-object name")
        seen_names.add(kernel_name)
        if int(row["splitK"]) != 0:
            raise ValueError(f"{kernel_name} advertises unsupported split-K")
        if str(row["pack_layout"]).strip() != PACK_LAYOUT:
            continue
        tile_m = int(row["tile_M"])
        tile_n = int(row["tile_N"])
        block_size = int(row["block_size"])
        if tile_m <= 0 or tile_n <= 0:
            raise ValueError(f"{kernel_name} has invalid tile dimensions")
        if block_size <= 0:
            raise ValueError(f"{kernel_name} has an invalid workgroup size")
        if not os.path.exists(os.path.join(os.path.dirname(manifest), co_name)):
            raise FileNotFoundError(f"missing A6W6 code object: {co_name}")
        swizzle_max_m = int(row["swizzle_max_M"])
        swizzle_max_n = int(row["swizzle_max_N"])
        swizzle_max_k = int(row["swizzle_max_K"])
        if min(swizzle_max_m, swizzle_max_n, swizzle_max_k) < 0 or (
            swizzle_max_k > 0 and (swizzle_max_m <= 0 or swizzle_max_n <= 0)
        ):
            raise ValueError(f"{kernel_name} has invalid swizzle bounds")
        candidates.append(
            {
                "kernel_id": int(kernel_id),
                "kernel_name": kernel_name,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "swizzle_max_m": swizzle_max_m,
                "swizzle_max_n": swizzle_max_n,
                "swizzle_max_k": swizzle_max_k,
            }
        )
    if not candidates:
        raise RuntimeError(f"no compatible A6W6 candidates in {manifest}")
    return candidates


def candidate_supports_shape(
    candidate: F6GemmCandidate, M: int, N: int, K: int
) -> bool:
    """Return whether a kernel's compile-time swizzle bounds cover the launch."""
    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    max_k = candidate["swizzle_max_k"]
    if max_k <= 0 or padK > max_k:
        return True
    return padM <= candidate["swizzle_max_m"] and padN <= candidate["swizzle_max_n"]


def generate_data(
    M: int,
    N: int,
    K: int,
    seed: int,
    device: str = "cuda",
    dtype: torch.dtype = dtypes.bf16,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((N, K), dtype=dtype, device=device)

    x_packed, x_scales = aiter.quant_mxfp6_gemm(x)
    w_packed, w_scales = aiter.quant_mxfp6_gemm(w)
    out = torch.empty((_ceil(M, 256), _ceil(N, 256)), dtype=dtype, device=device)
    # Candidate kernels share the same quantized math. Use the safe production
    # fallback as the tuning reference so large diffusion sweeps do not allocate
    # multi-gigabyte fp32 dequantized operands/results. Independent numerical
    # validation remains in op_tests/test_gemm_a6w6.py.
    from aiter.ops.gemm_op_a6w6 import _default_gemm_a6w6_kernel

    reference_out = torch.empty_like(out)
    aiter.gemm_a6w6_asm(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        reference_out,
        _ceil(K, 128),
        _default_gemm_a6w6_kernel(M, N, K),
    )
    ref = reference_out[:M, :N].clone()
    return {
        "x_packed": x_packed,
        "w_packed": w_packed,
        "x_scales": x_scales,
        "w_scales": w_scales,
        "out": out,
        "ref": ref,
    }


def return_reference(ref: torch.Tensor) -> torch.Tensor:
    return ref


def run_gemm_a6w6_asm(
    x_packed: torch.Tensor,
    w_packed: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out: torch.Tensor,
    M: int,
    N: int,
    K: int,
    kernel_name: str,
) -> torch.Tensor:
    aiter.gemm_a6w6_asm(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        out,
        _ceil(K, 128),
        kernel_name,
    )
    return out[:M, :N]


def choose_guarded_kernel(
    default_kernel: str,
    candidate_kernel: str,
    paired_speedups: list[float],
    min_gain_pct: float,
    exact_match: bool,
) -> tuple[str, float]:
    """Keep a candidate only when paired validation clears the gain threshold."""
    if candidate_kernel == default_kernel:
        return default_kernel, 0.0
    valid_speedups = []
    for value in paired_speedups:
        speedup = float(value)
        if math.isfinite(speedup) and speedup > 0:
            valid_speedups.append(speedup)
    if (
        not exact_match
        or not valid_speedups
        or len(valid_speedups) != len(paired_speedups)
    ):
        return default_kernel, float("-inf")
    gain_pct = (statistics.median(valid_speedups) - 1.0) * 100.0
    all_pairs_win = all(speedup > 1.0 for speedup in valid_speedups)
    selected = (
        candidate_kernel
        if gain_pct >= min_gain_pct and all_pairs_win
        else default_kernel
    )
    return selected, gain_pct


def _event_time_us(launch: Callable[[], Any], iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


@torch.no_grad()
def benchmark_candidate_gain(
    M: int,
    N: int,
    K: int,
    default_kernel: str,
    candidate_kernel: str,
    warmup: int,
    iters: int,
    reps: int,
    seed: int = 1,
) -> dict[str, Any]:
    """Pair a selected candidate against the safe default on identical operands."""
    torch.manual_seed(seed)
    x = torch.randn((M, K), dtype=dtypes.bf16, device="cuda")
    x_packed, x_scales = aiter.quant_mxfp6_gemm(x)
    del x
    torch.cuda.empty_cache()

    w = torch.randn((N, K), dtype=dtypes.bf16, device="cuda")
    w_packed, w_scales = aiter.quant_mxfp6_gemm(w)
    del w
    torch.cuda.empty_cache()

    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    default_out = torch.empty((padM, padN), dtype=dtypes.bf16, device="cuda")
    candidate_out = torch.empty_like(default_out)

    def launch_default():
        return aiter.gemm_a6w6_asm(
            x_packed,
            w_packed,
            x_scales,
            w_scales,
            default_out,
            padK,
            default_kernel,
        )

    def launch_candidate():
        return aiter.gemm_a6w6_asm(
            x_packed,
            w_packed,
            x_scales,
            w_scales,
            candidate_out,
            padK,
            candidate_kernel,
        )

    launch_default()
    launch_candidate()
    torch.cuda.synchronize()
    exact_match = bool(torch.equal(default_out[:M, :N], candidate_out[:M, :N]))

    default_times = []
    candidate_times = []
    paired_speedups = []
    for rep in range(reps):
        for _ in range(warmup):
            launch_default()
            launch_candidate()
        torch.cuda.synchronize()
        if rep % 2 == 0:
            default_us = _event_time_us(launch_default, iters)
            candidate_us = _event_time_us(launch_candidate, iters)
        else:
            candidate_us = _event_time_us(launch_candidate, iters)
            default_us = _event_time_us(launch_default, iters)
        default_times.append(default_us)
        candidate_times.append(candidate_us)
        paired_speedups.append(default_us / candidate_us)

    return {
        "default_us": float(statistics.median(default_times)),
        "candidate_us": float(statistics.median(candidate_times)),
        "paired_speedups": paired_speedups,
        "exact_match": exact_match,
    }


class GemmA6W6Tuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_GEMM_A6W6,
        "untune_file": "aiter/configs/a6w6_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A6W6",
        "min_gain_pct": 0.5,
        "gain_guard_warmup": 50,
        "gain_guard_iters": 300,
        "gain_guard_reps": 7,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _disable_core_dumps()
        self.candidates = load_f6gemm_candidates()
        self.candidates_by_id = {
            candidate["kernel_id"]: candidate for candidate in self.candidates
        }
        super().__init__(*args, **kwargs)

    def _setup_specific_arguments(self):
        defaults = self.get_arg_defaults()
        self.parser.add_argument(
            "--min-gain-pct",
            type=float,
            default=defaults["min_gain_pct"],
            help="Keep a non-default kernel only when paired validation beats "
            "the safe default by at least this percentage and every pair wins "
            "(default: 0.5).",
        )
        self.parser.add_argument(
            "--gain-guard-warmup",
            type=int,
            default=defaults["gain_guard_warmup"],
            help="Interleaved warmup pairs for minimum-gain validation.",
        )
        self.parser.add_argument(
            "--gain-guard-iters",
            type=int,
            default=defaults["gain_guard_iters"],
            help="Timed launches per kernel for each gain-validation repeat.",
        )
        self.parser.add_argument(
            "--gain-guard-reps",
            type=int,
            default=defaults["gain_guard_reps"],
            help="Paired timing repeats used by the minimum-gain guard.",
        )
        self.parser.add_argument(
            "--disable-gain-guard",
            action="store_true",
            help="Disable paired validation against the safe default.",
        )

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a6w6 import clear_gemm_a6w6_config_cache

        clear_gemm_a6w6_config_cache()

    def getKernelName(self, kernel_id):
        candidate = self.candidates_by_id.get(kernel_id)
        return None if candidate is None else candidate["kernel_name"]

    def calculate(self, results, bpes=(0.75, 0.75, 2)):
        return super().calculate(results, bpes=bpes)

    def _apply_min_gain_guard(self, resultdf, args):
        if resultdf.empty or getattr(args, "disable_gain_guard", False):
            return resultdf

        min_gain_pct = float(args.min_gain_pct)
        warmup = int(args.gain_guard_warmup)
        iters = int(args.gain_guard_iters)
        reps = int(args.gain_guard_reps)
        if min_gain_pct < 0 or not math.isfinite(min_gain_pct):
            raise ValueError("--min-gain-pct must be finite and non-negative")
        if warmup < 0 or iters <= 0 or reps <= 0:
            raise ValueError(
                "gain-guard warmup must be non-negative and iters/reps positive"
            )

        from aiter.ops.gemm_op_a6w6 import _default_gemm_a6w6_kernel

        candidates_by_name = {
            candidate["kernel_name"]: candidate for candidate in self.candidates
        }
        guarded = resultdf.copy()
        for index, row in guarded.iterrows():
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            candidate_kernel = str(row["kernelName"])
            default_kernel = _default_gemm_a6w6_kernel(M, N, K)
            if candidate_kernel == default_kernel:
                continue

            validation = None
            try:
                validation = benchmark_candidate_gain(
                    M,
                    N,
                    K,
                    default_kernel,
                    candidate_kernel,
                    warmup,
                    iters,
                    reps,
                )
                selected_kernel, gain_pct = choose_guarded_kernel(
                    default_kernel,
                    candidate_kernel,
                    validation["paired_speedups"],
                    min_gain_pct,
                    validation["exact_match"],
                )
            except Exception as error:  # noqa: BLE001
                selected_kernel = default_kernel
                gain_pct = float("-inf")
                logger.warning(
                    "A6W6 gain validation failed for M=%s N=%s K=%s: %s; "
                    "falling back to %s",
                    M,
                    N,
                    K,
                    error,
                    default_kernel,
                )

            selected = candidates_by_name[selected_kernel]
            guarded.at[index, "kernelId"] = selected["kernel_id"]
            guarded.at[index, "kernelName"] = selected_kernel
            guarded.at[index, "splitK"] = 0
            if validation is not None:
                selected_us = (
                    validation["candidate_us"]
                    if selected_kernel == candidate_kernel
                    else validation["default_us"]
                )
                guarded.at[index, "us"] = round(selected_us, 4)
                err_ratio = (
                    float(row["errRatio"])
                    if selected_kernel == candidate_kernel
                    else 0.0
                )
                info = (
                    (
                        (
                            str(row["gfx"]),
                            int(row["cu_num"]),
                            M,
                            N,
                            K,
                        ),
                        selected["kernel_id"],
                        0,
                        selected_kernel,
                    ),
                    selected_us,
                    err_ratio,
                )
                tflops, bw = self.calculate(info)
                guarded.at[index, "tflops"] = tflops
                guarded.at[index, "bw"] = bw
                if selected_kernel == default_kernel:
                    guarded.at[index, "errRatio"] = 0.0

            action = "KEEP" if selected_kernel == candidate_kernel else "FALLBACK"
            all_pairs_win = bool(
                validation is not None
                and all(
                    float(speedup) > 1.0 for speedup in validation["paired_speedups"]
                )
            )
            print(
                "[gain-guard] "
                f"M={M} N={N} K={K} candidate={candidate_kernel} "
                f"gain={gain_pct:+.3f}% threshold={min_gain_pct:.3f}% "
                f"all_pairs_win={all_pairs_win} "
                f"action={action} selected={selected_kernel}",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        return guarded

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        result = super().post_process(rets, args, topk, fast_mode)
        if isinstance(result, pd.DataFrame) and topk == 1 and not fast_mode:
            return self._apply_min_gain_guard(result, args)
        return result

    def run_config(self, args):
        from aiter.ops.gemm_op_a6w6 import gemm_a6w6
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        for _, row in self.untunedf.iterrows():
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            shape_str = f"({M}, {N}, {K})"
            allowed_err_ratio, allowed_desc = self._get_run_config_err_ratio_limit(
                row, args
            )
            try:
                data = generate_data(M, N, K, seed=0)
                out, us = run_perftest(
                    gemm_a6w6,
                    data["x_packed"],
                    data["w_packed"],
                    data["x_scales"],
                    data["w_scales"],
                    M,
                    N,
                    K,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                err_ratio = checkAllclose(
                    data["ref"],
                    out,
                    msg=f"run_config {shape_str}",
                    catastrophic_check=True,
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as error:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{error}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(self, untunedf, tunedf, args):
        del tunedf
        if args.splitK:
            raise ValueError("A6W6 does not support split-K")
        if self.get_gfx() != "gfx950":
            raise RuntimeError(f"A6W6 tuning requires gfx950, got {self.get_gfx()}")

        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        tasks = []
        tasks_in_data = []
        gemm_keys = [
            "x_packed",
            "w_packed",
            "x_scales",
            "w_scales",
            "out",
        ]
        seed = 0
        for _, row in untunedf.iterrows():
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            compatible_candidates = [
                candidate
                for candidate in self.candidates
                if candidate_supports_shape(candidate, M, N, K)
            ]
            if not compatible_candidates:
                raise RuntimeError(f"no safe A6W6 candidate for {(M, N, K)}")
            for candidate in compatible_candidates:
                kernel_id = candidate["kernel_id"]
                kernel_name = candidate["kernel_name"]
                info = ((gfx, cu_num, M, N, K), kernel_id, 0, kernel_name)
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a6w6_asm,
                        (gemm_keys, M, N, K, kernel_name),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        return_reference,
                        (["ref"],),
                        {},
                        None,
                        1e-2,
                        1e-2,
                        None,
                        None,
                        ("out",),
                    )
                )
            tasks_in_data.append((len(compatible_candidates), ()))

        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_in_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    tuner = GemmA6W6Tuner(
        "GemmA6W6Tuner",
        description="Tune gfx950 MXFP6 A6W6 assembly kernels",
    )
    tuner.run(tuner.parse_args(), False)
