# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune the gfx942 A16W8 MXFP8 assembly kernels (BF16 activations, MXFP8 weights) by shape.

Every kernel of hsa/gfx942/a16w8gemm/a16w8gemm_mxfp8.csv that admits the shape
(M <= 16 * tr, N % (16 * tn) == 0, split-K buffers within the op's workspace) is
a candidate; split-K kernels are tried at every --splitk-values split.

1. The common tuner (mp_tuner) times every candidate with run_perftest, operands
   rotated through cache-sized copies, and checks it against a float64
   reference.
2. Decode runs these GEMMs in CUDA graphs, where per-launch costs that the
   profiler's kernel time leaves out differ between kernels. The --graph-topk
   fastest candidates and BF16 (F.linear on the dequantized weights, the GEMM
   the kernel replaces) are therefore timed again in CUDA graphs, interleaved
   over --graph-reps rounds, with the weights rotated over more than 1 GiB of
   copies so the 256 MB Infinity Cache does not serve them. The candidate with
   the lowest median graph time is the shape's kernel.
3. A row is written only when that kernel beats BF16 (median paired gain above
   --min-gain-pct). Other shapes are left out of the tuned CSV (and removed from
   it when re-tuned), so is_gemm_a16w8_mxfp8_tuned() stays False there and
   callers keep BF16.
"""

import gc
import math
import os
import statistics
import sys
from collections.abc import Callable
from typing import Any, ClassVar, TypedDict

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_GEMM_A16W8_MXFP8_ASM, AITER_META_DIR
from aiter.jit.utils.chip_info import get_cu_num as get_runtime_cu_num
from aiter.ops.gemm_op_a16w8_mxfp8 import _MX_COUNTERS, _MX_WORKSPACE_BYTES
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

_GFX = "gfx942"
_REGISTRY = os.path.join("hsa", _GFX, "a16w8gemm", "a16w8gemm_mxfp8.csv")
_REGISTRY_COLUMNS = frozenset({"knl_name", "co_name", "tr", "tn", "nw", "sk"})
_GROUP = 32
_MAX_M = 64
_MAX_SPLITK = 8
_SPLITK_VALUES = "2,3,4,6,8"
# Bytes per element of A (bf16), B (fp8 plus one exponent byte per 32) and out (bf16).
_BPES = (2.0, 1.0 + 1.0 / _GROUP, 2.0)
_ROTATE_BYTES = 1 << 30


class A16W8Mxfp8Candidate(TypedDict):
    kernel_id: int
    kernel_name: str
    tr: int
    tn: int
    sk: bool


def _registry_path() -> str:
    return os.path.join(AITER_META_DIR, _REGISTRY)


def load_a16w8_mxfp8_candidates() -> list[A16W8Mxfp8Candidate]:
    """Registered kernels; kernelId is the row index in the registry csv."""
    registry = _registry_path()
    configs = pd.read_csv(registry)
    missing = _REGISTRY_COLUMNS - set(configs.columns)
    if missing:
        raise ValueError(f"{registry} is missing columns: {sorted(missing)}")
    candidates: list[A16W8Mxfp8Candidate] = []
    seen: set[str] = set()
    for kernel_id, row in configs.iterrows():
        kernel_name = str(row["knl_name"]).strip()
        co_name = str(row["co_name"]).strip()
        if not kernel_name or kernel_name in seen:
            raise ValueError(f"invalid or duplicate kernel name: {kernel_name!r}")
        seen.add(kernel_name)
        if not os.path.exists(os.path.join(os.path.dirname(registry), co_name)):
            raise FileNotFoundError(f"missing A16W8 MXFP8 code object: {co_name}")
        tr, tn = int(row["tr"]), int(row["tn"])
        if tr <= 0 or tn <= 0 or int(row["nw"]) <= 0:
            raise ValueError(f"{kernel_name} has invalid tile parameters")
        candidates.append(
            {
                "kernel_id": int(kernel_id),
                "kernel_name": kernel_name,
                "tr": tr,
                "tn": tn,
                "sk": bool(int(row["sk"])),
            }
        )
    if not candidates:
        raise RuntimeError(f"no kernels in {registry}")
    return candidates


def candidate_supports_shape(
    candidate: A16W8Mxfp8Candidate, M: int, N: int, splitK: int
) -> bool:
    """The launcher's limits, and split-K buffers that fit the op's workspace."""
    if M > 16 * candidate["tr"] or N % (16 * candidate["tn"]):
        return False
    if not candidate["sk"]:
        return True
    tiles = N // 16 * candidate["tr"]  # one 16 x 16 tile per row and column block
    return tiles <= _MX_COUNTERS and splitK * tiles * 1024 <= _MX_WORKSPACE_BYTES


def _make_operands(
    M: int, N: int, K: int, seed: int, device: Any = "cuda"
) -> dict[str, torch.Tensor]:
    """Normal random weights quantized to OCP MXFP8 (e4m3fn, UE8M0 per 32 along K)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    groups = K // _GROUP
    fn_max = torch.finfo(torch.float8_e4m3fn).max
    a = torch.randn(M, K, generator=gen, device=device).to(dtypes.bf16)
    w = torch.randn(N, K, generator=gen, device=device) * K**-0.5
    amax = w.view(N, groups, _GROUP).abs().amax(-1).clamp_min(2.0**-100)
    e = torch.floor(torch.log2(amax)) - 8  # e4m3fn's largest power of two is 2^8
    q = w.view(N, groups, _GROUP) / torch.exp2(e)[..., None]
    w_fn = q.clamp(-fn_max, fn_max).to(torch.float8_e4m3fn).view(N, K)
    w_scale = (e + 127).to(torch.uint8)
    dequant = (
        w_fn.double().view(N, groups, _GROUP)
        * torch.exp2(w_scale.double() - 127)[..., None]
    ).view(N, K)
    b, b_scale = aiter.gemm_a16w8_mxfp8_prepare_weight(w_fn, w_scale)
    return {"a": a, "b": b, "b_scale": b_scale, "dequant": dequant}


def generate_data(
    M: int, N: int, K: int, seed: int, device: Any = "cuda"
) -> dict[str, torch.Tensor]:
    """Prepared operands, an output buffer and the float64 reference rounded to BF16."""
    ops = _make_operands(M, N, K, seed, device)
    ref = (ops["a"].double() @ ops["dequant"].t()).to(dtypes.bf16)
    return {
        "a": ops["a"],
        "b": ops["b"],
        "b_scale": ops["b_scale"],
        "out": torch.empty(M, N, dtype=dtypes.bf16, device=device),
        "ref": ref,
    }


def return_reference(ref: torch.Tensor) -> torch.Tensor:
    return ref


def run_gemm_a16w8_mxfp8_asm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    kernel_name: str,
    splitK: int,
) -> torch.Tensor:
    aiter.gemm_a16w8_mxfp8_asm(
        a, b, b_scale, out, kernelName=kernel_name, splitK=splitK
    )
    return out


def _capture(fn: Callable[[int], Any], iters: int) -> torch.cuda.CUDAGraph:
    """`iters` calls of fn(i) in one CUDA graph, after a warmup on a side stream."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            fn(i)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _replay_us(graph: torch.cuda.CUDAGraph, iters: int, replays: int = 3) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / (replays * iters))


@torch.no_grad()
def benchmark_in_graphs(
    M: int,
    N: int,
    K: int,
    candidates: list[tuple[str, int]],
    iters: int,
    reps: int,
    seed: int = 1,
) -> tuple[list[list[float]], list[float]]:
    """us per call of each (kernelName, splitK) and of BF16 F.linear, per round.

    Each arm is `iters` calls captured in one CUDA graph that walk through
    copies of the weights (more than 1 GiB in total). The arms are timed in
    `reps` rounds, in a rotated order each round.
    """
    ops = _make_operands(M, N, K, seed)
    a = ops["a"]
    w_bf16 = ops.pop("dequant").to(dtypes.bf16)
    mx_copies = max(2, _ROTATE_BYTES // (N * K * 33 // 32) + 1)
    bf_copies = max(2, _ROTATE_BYTES // (N * K * 2) + 1)
    mx_b = [ops["b"].clone() for _ in range(mx_copies)]
    mx_s = [ops["b_scale"].clone() for _ in range(mx_copies)]
    bf_w = [w_bf16.clone() for _ in range(bf_copies)]
    del ops, w_bf16
    out = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")

    def asm_arm(kernel_name: str, splitK: int) -> Callable[[int], None]:
        def run(i: int) -> None:
            aiter.gemm_a16w8_mxfp8_asm(
                a,
                mx_b[i % mx_copies],
                mx_s[i % mx_copies],
                out,
                kernelName=kernel_name,
                splitK=splitK,
            )

        return run

    def bf16_arm(i: int) -> torch.Tensor:
        return F.linear(a, bf_w[i % bf_copies])

    arms = [asm_arm(name, splitK) for name, splitK in candidates] + [bf16_arm]
    graphs = [_capture(fn, iters) for fn in arms]
    times: list[list[float]] = [[] for _ in arms]
    for rep in range(reps):
        order = list(range(len(arms)))
        order = order[rep % len(order) :] + order[: rep % len(order)]
        for i in order:
            times[i].append(_replay_us(graphs[i], iters))
    return times[:-1], times[-1]


def _key(values) -> tuple[str, ...]:
    return tuple(str(v) for v in values)


class GemmA16W8Mxfp8Tuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_GEMM_A16W8_MXFP8_ASM,
        "untune_file": "aiter/configs/a16w8_mxfp8_asm_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A16W8_MXFP8_ASM",
        "splitk_values": _SPLITK_VALUES,
        "graph_topk": 8,
        "graph_iters": 200,
        "graph_reps": 5,
        "min_gain_pct": 0.0,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.candidates = load_a16w8_mxfp8_candidates()
        self.candidates_by_id = {c["kernel_id"]: c for c in self.candidates}
        self._bf16_preferred: set[tuple[str, ...]] = set()
        super().__init__(*args, **kwargs)

    def _setup_specific_arguments(self):
        defaults = self.get_arg_defaults()
        self.parser.add_argument(
            "--splitk-values",
            default=defaults["splitk_values"],
            help="Comma-separated split counts tried for the split-K kernels "
            f"(1 to {_MAX_SPLITK}; default {_SPLITK_VALUES}).",
        )
        self.parser.add_argument(
            "--graph-topk",
            type=int,
            default=defaults["graph_topk"],
            help="Fastest candidates per shape (by kernel time) that are timed "
            "again in CUDA graphs against BF16 (default 8).",
        )
        self.parser.add_argument(
            "--graph-iters",
            type=int,
            default=defaults["graph_iters"],
            help="Calls per CUDA graph in the graph timing (default 200).",
        )
        self.parser.add_argument(
            "--graph-reps",
            type=int,
            default=defaults["graph_reps"],
            help="Interleaved rounds of the graph timing; medians are used "
            "(default 5).",
        )
        self.parser.add_argument(
            "--min-gain-pct",
            type=float,
            default=defaults["min_gain_pct"],
            help="Write a shape only when its kernel's median paired gain over "
            "BF16 exceeds this percentage (default 0: strictly faster).",
        )
        self.parser.add_argument(
            "--disable-bf16-guard",
            action="store_true",
            help="Skip the graph timing and the BF16 comparison and write the "
            "fastest kernel by kernel time for every shape (diagnostics only).",
        )

    def _splitk_values(self, args) -> tuple[int, ...]:
        values = tuple(int(v) for v in str(args.splitk_values).split(",") if v.strip())
        if not values or any(not 1 <= v <= _MAX_SPLITK for v in values):
            raise ValueError(f"--splitk-values must be in 1..{_MAX_SPLITK}")
        return values

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a16w8_mxfp8 import clear_gemm_a16w8_mxfp8_config_cache

        clear_gemm_a16w8_mxfp8_config_cache()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        from aiter.jit import core as jit_core

        type(jit_core.AITER_CONFIGS).get_config_file.cache_clear()
        self._clear_op_caches()

    def get_cu_num(self):
        """The CU count the op's tuned lookup uses (honours CU_NUM)."""
        return get_runtime_cu_num()

    def pre_process(self, args):
        if args.run_config is True:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)
            self.untunedf["gfx"] = self.get_gfx()
            self.untunedf["cu_num"] = self.get_cu_num()
            self.untunedf = self.untunedf[self.keys]
            self.tunedf = self.get_tuned_gemm_list(args.tune_file)
            return
        super().pre_process(args)

    def getKernelName(self, kernel_id):
        candidate = self.candidates_by_id.get(kernel_id)
        return None if candidate is None else candidate["kernel_name"]

    def calculate(self, results, bpes=_BPES):
        return super().calculate(results, bpes=bpes)

    def _graph_select(self, ranked: pd.DataFrame, args) -> pd.DataFrame:
        """Pick each shape's kernel by graph time and keep it only if it beats BF16."""
        iters, reps = int(args.graph_iters), int(args.graph_reps)
        min_gain_pct = float(args.min_gain_pct)
        if iters <= 0 or reps <= 0 or not math.isfinite(min_gain_pct):
            raise ValueError("--graph-iters / --graph-reps must be positive")
        selected = []
        for key, rows in ranked.groupby(self.keys, sort=False):
            group = rows[rows["us"] != self.INVALID_TIME]
            if group.empty:  # no candidate passed: keep the failure row
                selected.append(rows.iloc[:1])
                continue
            M, N, K = (int(v) for v in key[2:5])
            cands = [
                (str(r["kernelName"]), int(r["splitK"])) for _, r in group.iterrows()
            ]
            try:
                asm_times, bf16_times = benchmark_in_graphs(M, N, K, cands, iters, reps)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "A16W8 MXFP8 graph timing failed for M=%s N=%s K=%s: %s",
                    M,
                    N,
                    K,
                    error,
                )
                failed = group.iloc[:1].copy()
                failed["us"] = self.INVALID_TIME
                selected.append(failed)
                continue
            medians = [statistics.median(t) for t in asm_times]
            best = min(range(len(cands)), key=medians.__getitem__)
            gain_pct = (
                statistics.median(b / a for a, b in zip(asm_times[best], bf16_times))
                - 1.0
            ) * 100.0
            bf16_us = statistics.median(bf16_times)
            for i, (name, splitK) in enumerate(cands):
                print(
                    f"[graph] M={M} N={N} K={K} {name} splitK={splitK} "
                    f"kernel={float(group.iloc[i]['us']):.2f}us graph={medians[i]:.2f}us",
                    flush=True,
                )
            keep = gain_pct > min_gain_pct
            print(
                "[bf16-guard] "
                f"M={M} N={N} K={K} kernel={cands[best][0]} splitK={cands[best][1]} "
                f"asm={medians[best]:.2f}us bf16={bf16_us:.2f}us "
                f"gain={gain_pct:+.2f}% threshold={min_gain_pct:.2f}% "
                f"action={'KEEP' if keep else 'DROP (BF16 faster)'}",
                flush=True,
            )
            if not keep:
                self._bf16_preferred.add(_key(key))
                continue
            row = group.iloc[[best]].copy()
            us = round(medians[best], 2)
            info = (
                (
                    (key[0], int(key[1]), M, N, K),
                    int(row["kernelId"].iloc[0]),
                    cands[best][1],
                    cands[best][0],
                ),
                us,
                float(row["errRatio"].iloc[0]),
            )
            tflops, bw = self.calculate(info)
            row["us"], row["tflops"], row["bw"] = us, tflops, bw
            selected.append(row)
            gc.collect()
            torch.cuda.empty_cache()
        if not selected:
            return ranked.iloc[0:0]
        return pd.concat(selected, ignore_index=True)

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        if topk != 1 or fast_mode or args.disable_bf16_guard:
            return super().post_process(rets, args, topk, fast_mode)
        if args.graph_topk <= 0:
            raise ValueError("--graph-topk must be positive")
        ranked = super().post_process(rets, args, args.graph_topk, fast_mode)
        self.topk = 1
        return self._graph_select(ranked, args)

    def result_to_csv(self, resultdf, file, concat=False):
        super().result_to_csv(resultdf, file, concat)
        if not self._bf16_preferred or not os.path.exists(file):
            return
        # Re-tuned shapes that BF16 now wins must not keep an older row.
        tuned = pd.read_csv(file)
        stale = tuned[self.keys].apply(_key, axis=1).isin(self._bf16_preferred)
        if stale.any():
            tuned[~stale].to_csv(file, index=False, na_rep="Null")

    def tune_summary(self, status):
        if self._bf16_preferred and self.untunedf is not None:
            logger.info("BF16 is faster for these shapes; they have no tuned row:")
            for key in sorted(self._bf16_preferred):
                print(dict(zip(self.keys, key)), flush=True)
            keep = (
                ~self.untunedf[self.keys].apply(_key, axis=1).isin(self._bf16_preferred)
            )
            self.untunedf = self.untunedf[keep.values].reset_index(drop=True)
            if self.untunedf.empty:
                if not self.failed.empty:
                    print(self.failed, flush=True)
                    sys.exit(1)
                return
        super().tune_summary(status)

    def run_config(self, args):
        from aiter.ops.gemm_op_a16w8_mxfp8 import _mx_default_kernel
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
                kwargs = (
                    {"kernelName": _mx_default_kernel(M, N)}
                    if args.run_config is True
                    else {}
                )
                out, us = run_perftest(
                    aiter.gemm_a16w8_mxfp8_asm,
                    data["a"],
                    data["b"],
                    data["b_scale"],
                    data["out"],
                    **kwargs,
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
        if self.get_gfx() != _GFX:
            raise RuntimeError(
                f"A16W8 MXFP8 tuning requires {_GFX}, got {self.get_gfx()}"
            )
        splitk_values = self._splitk_values(args)
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        tasks = []
        tasks_in_data = []
        seed = 0
        for _, row in untunedf.iterrows():
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            if not 0 < M <= _MAX_M or K % _GROUP or N % 16:
                raise ValueError(
                    f"unsupported shape {(M, N, K)}: needs 1 <= M <= {_MAX_M}, "
                    f"N % 16 == 0, K % {_GROUP} == 0"
                )
            candidates = [
                (candidate, splitK)
                for candidate in self.candidates
                for splitK in (splitk_values if candidate["sk"] else (0,))
                if candidate_supports_shape(candidate, M, N, splitK)
            ]
            for candidate, splitK in candidates:
                info = (
                    (gfx, cu_num, M, N, K),
                    candidate["kernel_id"],
                    splitK,
                    candidate["kernel_name"],
                )
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a16w8_mxfp8_asm,
                        (
                            ["a", "b", "b_scale", "out"],
                            candidate["kernel_name"],
                            splitK,
                        ),
                        {"num_warmup": args.warmup, "num_iters": args.iters},
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
            tasks_in_data.append((len(candidates), ()))
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
    tuner = GemmA16W8Mxfp8Tuner(
        "GemmA16W8Mxfp8Tuner",
        description="Tune the gfx942 A16W8 MXFP8 assembly kernels",
    )
    tuner.run(tuner.parse_args(), False)
