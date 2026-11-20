# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gfx950 MXFP6 A6W6 kernels by GEMM shape.

The candidates are the FlyDSL 4-wave kernel's variants (tile order and K-tile
schedule) and the registered assembly kernels. A row keeps the candidate that paired
validation shows the most faster than the untuned default (the FlyDSL kernel's
default variant), by at least the minimum gain; otherwise the default.
"""

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
_FLYDSL_PREFIX = "flydsl_a6w6_"
# FlyDSL candidates: M-tile swizzle group by MFMA block, barrier after MFMA 8. Their
# kernelIds start above the assembly manifest's row indices.
FLYDSL_KERNEL_ID_BASE = 1000
FLYDSL_VARIANTS = tuple(
    {"xcd_swizzle": group, "mblk_n": mblk_n, "bar": 8}
    for mblk_n in (2, 4)
    for group in (1, 2, 3, 4, 8)
)
# Profiling times each candidate once, which cannot order kernels within about 2 % of
# each other, so the gain guard re-times this many of the fastest per shape, and the
# fastest assembly kernel wherever that is not among them.
GUARD_CANDIDATES = 4
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
        "exact_M",
        "exact_N",
        "exact_K",
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
    exact_m: int
    exact_n: int
    exact_k: int


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
        exact_m = int(row["exact_M"])
        exact_n = int(row["exact_N"])
        exact_k = int(row["exact_K"])
        if (exact_m, exact_n, exact_k) != (0, 0, 0) and min(
            exact_m, exact_n, exact_k
        ) <= 0:
            raise ValueError(f"{kernel_name} has an incomplete exact shape")
        candidates.append(
            {
                "kernel_id": int(kernel_id),
                "kernel_name": kernel_name,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "swizzle_max_m": swizzle_max_m,
                "swizzle_max_n": swizzle_max_n,
                "swizzle_max_k": swizzle_max_k,
                "exact_m": exact_m,
                "exact_n": exact_n,
                "exact_k": exact_k,
            }
        )
    if not candidates:
        raise RuntimeError(f"no compatible A6W6 candidates in {manifest}")
    return candidates


def flydsl_candidates() -> list[tuple[int, str]]:
    """(kernelId, kernelName) of each FlyDSL variant; every one runs any K."""
    from aiter.ops.flydsl.gemm_a6w6_4wave import make_kernel_name

    return [
        (FLYDSL_KERNEL_ID_BASE + index, make_kernel_name(**variant))
        for index, variant in enumerate(FLYDSL_VARIANTS)
    ]


def flydsl_kernel_id(kernel_name: str) -> int | None:
    """The kernelId of a FlyDSL variant's name, or None for any other name."""
    from aiter.ops.flydsl.gemm_a6w6_4wave import parse_kernel_name

    config = parse_kernel_name(kernel_name)
    if config is None:
        return None
    for index, variant in enumerate(FLYDSL_VARIANTS):
        if variant == config:
            return FLYDSL_KERNEL_ID_BASE + index
    return None


def candidate_supports_shape(
    candidate: F6GemmCandidate, M: int, N: int, K: int
) -> bool:
    """Return whether a kernel's exact-shape and swizzle contracts cover the launch."""
    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    if candidate["exact_k"] > 0:
        return (padM, padN, padK) == (
            candidate["exact_m"],
            candidate["exact_n"],
            candidate["exact_k"],
        )
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
    # Candidate kernels share the same quantized math. Use the safe assembly
    # fallback, an independent implementation, as the tuning reference so large
    # diffusion sweeps do not allocate multi-gigabyte fp32 dequantized
    # operands/results. Independent numerical validation remains in
    # op_tests/test_gemm_a6w6.py.
    from aiter.ops.gemm_op_a6w6 import _default_gemm_a6w6_asm_kernel

    reference_out = torch.empty_like(out)
    aiter.gemm_a6w6_asm(
        x_packed,
        w_packed,
        x_scales,
        w_scales,
        reference_out,
        _ceil(K, 128),
        _default_gemm_a6w6_asm_kernel(M, N, K),
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


def _launch(x_packed, w_packed, x_scales, w_scales, out, K, kernel_name):
    """Run one candidate by name: a FlyDSL variant or an assembly kernel."""
    if kernel_name.startswith(_FLYDSL_PREFIX):
        aiter.gemm_a6w6_flydsl_out(
            x_packed, w_packed, x_scales, w_scales, out, _ceil(K, 128), kernel_name
        )
    else:
        aiter.gemm_a6w6_asm(
            x_packed, w_packed, x_scales, w_scales, out, _ceil(K, 128), kernel_name
        )
    return out


def run_gemm_a6w6(
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
    _launch(x_packed, w_packed, x_scales, w_scales, out, K, kernel_name)
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
        return _launch(
            x_packed, w_packed, x_scales, w_scales, default_out, padK, default_kernel
        )

    def launch_candidate():
        return _launch(
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
    # The min-gain guard re-benchmarks the energy winner and keeps it only
    # for a latency speedup, so the published row would not be a power pick.
    SUPPORTS_POWER_SELECT = False

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
        # A FlyDSL variant's name carries K, so its tasks carry their names instead.
        candidate = self.candidates_by_id.get(kernel_id)
        return None if candidate is None else candidate["kernel_name"]

    def _kernel_id(self, kernel_name):
        flydsl_id = flydsl_kernel_id(kernel_name)
        if flydsl_id is not None:
            return flydsl_id
        for candidate in self.candidates:
            if candidate["kernel_name"] == kernel_name:
                return candidate["kernel_id"]
        raise KeyError(f"unknown A6W6 kernel {kernel_name!r}")

    def calculate(self, results, bpes=(0.75, 0.75, 2)):
        return super().calculate(results, bpes=bpes)

    def _apply_min_gain_guard(self, resultdf, args):
        """One row per shape, from its profiled candidates (fastest first): of the
        GUARD_CANDIDATES fastest and the fastest assembly kernel, the one paired
        validation shows the most faster than the default, by at least
        --min-gain-pct with every repeat a win; else the default."""
        if resultdf.empty:
            return resultdf
        if getattr(args, "disable_gain_guard", False):
            return resultdf.groupby(["M", "N", "K"], sort=False).head(1)

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

        rows = []
        for (M, N, K), group in resultdf.groupby(["M", "N", "K"], sort=False):
            M, N, K = int(M), int(N), int(K)
            default_kernel = _default_gemm_a6w6_kernel(M, N, K)
            is_asm = ~group["kernelName"].astype(str).str.startswith(_FLYDSL_PREFIX)
            retimed = pd.concat(
                [group.head(GUARD_CANDIDATES), group[is_asm].head(1)]
            ).drop_duplicates("kernelName")
            profiled_default = group.loc[group["kernelName"] == default_kernel, "us"]
            default_us = (
                float(profiled_default.iloc[0]) if len(profiled_default) else None
            )
            best = None
            for _, row in retimed.iterrows():
                candidate_kernel = str(row["kernelName"])
                if candidate_kernel == default_kernel:
                    continue
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
                    logger.warning(
                        "A6W6 gain validation failed for M=%s N=%s K=%s %s: %s",
                        M,
                        N,
                        K,
                        candidate_kernel,
                        error,
                    )
                    continue
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
                default_us = validation["default_us"]
                keep = selected_kernel == candidate_kernel
                print(
                    "[gain-guard] "
                    f"M={M} N={N} K={K} candidate={candidate_kernel} "
                    f"gain={gain_pct:+.3f}% threshold={min_gain_pct:.3f}% "
                    f"{'pass' if keep else 'fail'}",
                    flush=True,
                )
                if keep and (best is None or gain_pct > best[0]):
                    best = (gain_pct, row, validation["candidate_us"])
            if best is None:
                first = group.iloc[0]
                us = float(first["us"]) if default_us is None else default_us
                rows.append(self._guarded_row(first, default_kernel, us, 0.0))
            else:
                _, row, us = best
                rows.append(
                    self._guarded_row(
                        row, str(row["kernelName"]), us, float(row["errRatio"])
                    )
                )
            print(
                f"[gain-guard] M={M} N={N} K={K} selected={rows[-1]['kernelName']}",
                flush=True,
            )
        return pd.DataFrame(rows).reset_index(drop=True).infer_objects()

    def _guarded_row(self, row, kernel_name, us, err_ratio):
        """``row`` with ``kernel_name`` selected at ``us`` microseconds."""
        row = row.copy()
        kernel_id = self._kernel_id(kernel_name)
        key = (str(row["gfx"]), int(row["cu_num"]), *(int(row[k]) for k in "MNK"))
        row["kernelId"], row["splitK"], row["kernelName"] = kernel_id, 0, kernel_name
        row["us"], row["errRatio"] = round(us, 4), err_ratio
        row["tflops"], row["bw"] = self.calculate(
            ((key, kernel_id, 0, kernel_name), us, err_ratio)
        )
        return row

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        if topk != 1 or fast_mode:
            return super().post_process(rets, args, topk, fast_mode)
        rets = list(rets)
        # Every valid candidate of each shape, fastest first, for the gain guard.
        result = super().post_process(rets, args, len(rets), fast_mode)
        self.topk = 1
        if not isinstance(result, pd.DataFrame):
            return result
        return self._apply_min_gain_guard(result, args)

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
            compatible_candidates = flydsl_candidates() + [
                (candidate["kernel_id"], candidate["kernel_name"])
                for candidate in self.candidates
                if candidate_supports_shape(candidate, M, N, K)
            ]
            for kernel_id, kernel_name in compatible_candidates:
                info = ((gfx, cu_num, M, N, K), kernel_id, 0, kernel_name)
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a6w6,
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
        description="Tune gfx950 MXFP6 A6W6 kernels (FlyDSL variants and "
        "assembly kernels)",
    )
    tuner.run(tuner.parse_args(), False)
