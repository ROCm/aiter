# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a4w4 on gfx1250 (MXFP4, 1x32 e8m0 scales in the 32x4 layout).

Candidates are the FlyDSL compute 256x256 and general kernels plus the asm F4GEMM heuristic; the
fastest per shape becomes a flydsl or asm row of the a4w4_mxscale_bpreshuffle table.
Stage 1 times all compute configs and the general structures at 1x1 / no split-K, stage 2 adds
cluster shapes and split-K to the --top-structures best, stage 3 re-times the --confirm best and
asm in --confirm-rounds interleaved rounds (median).

    python3 csrc/gemm_a4w4_mxscale_bpreshuffle/gemm_a4w4_mxscale_bpreshuffle_tune.py \\
        -i <untuned.csv> -o <tuned.csv> --mp 1 --compile-workers 64
"""

import concurrent.futures
import functools
import multiprocessing
import os
import statistics
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import QuantType, dtypes, get_hip_quant, logger
from aiter.jit.core import AITER_ASM_DIR
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

TUNED_FILE = "aiter/configs/model_configs/dsr1_a4w4_mxscale_bpreshuffle_tuned_gemm.csv"
UNTUNED_FILE = (
    "aiter/configs/model_configs/dsr1_a4w4_mxscale_bpreshuffle_untuned_gemm.csv"
)
# The asm F4GEMM kernels for MXFP4, row-major A and a bf16 output (hsa/gfx1250/f4gemm).
ASM_KERNEL = "_ZN5aiter44f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_psE"
ASM_NPS_KERNEL = "_ZN5aiter41f4gemm_bf16_mxfp4_BpreShuffle_192x256_4x4E"


@functools.cache
def _asm_kernels(gfx):
    """knl_name of every kernel registered in hsa/<gfx>/f4gemm/f4gemm.csv."""
    try:
        df = pd.read_csv(f"{AITER_ASM_DIR}{gfx}/f4gemm/f4gemm.csv", comment="#")
    except OSError:
        return frozenset()
    return frozenset(df["knl_name"])


def asm_kernel_name(gfx, M, N, K):
    """The kernel gemm_mxfp4_asm runs for this shape, mirroring get_heuristic_kernel in
    csrc/py_itfs_cu/asm_f4gemm.cu: the 192x256 no-ps kernel, when registered, on whole 4x4
    clusters and at most 48 of them; else the persistent 256x256 one."""
    if (
        M % (192 * 4) == 0
        and N % (256 * 4) == 0
        and K % 256 == 0
        and M * N <= 48 * (192 * 4) * (256 * 4)
        and ASM_NPS_KERNEL in _asm_kernels(gfx)
    ):
        return ASM_NPS_KERNEL
    return ASM_KERNEL


def generate_data(m, n, k, seed, device="cuda"):
    """Random bf16 quantized to fp4; the shuffled operands hold the reference's bytes."""
    torch.manual_seed(seed)
    quant = get_hip_quant(QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtypes.bf16, device=device)
    w = torch.randn((n, k), dtype=dtypes.bf16, device=device)
    xq_raw, xs_raw = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    wq_raw, ws_raw = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    xq, xs = quant(x, quant_dtype=dtypes.fp4x2, scale_layout_m32k4=True)

    def dequant(q, s):
        scale = fp4_utils.e8m0_to_f32(s.view(torch.uint8)[: q.shape[0]])
        return fp4_utils.mxfp4_to_f32(q) * scale.repeat_interleave(32, dim=1)

    return {
        "x": xq,
        "weight_shuffle": shuffle_weight(wq_raw, layout=(16, 16)),
        "x_scale": xs,
        "w_scale": shuffle_mxfp8fp4_scale(ws_raw),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": dequant(xq_raw, xs_raw),
        "w_deq": dequant(wq_raw, ws_raw),
    }


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


def run_gemm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a4w4_mxscale_bpreshuffle_gfx1250,
    )

    return run_gemm_a4w4_mxscale_bpreshuffle_gfx1250(
        x, weight_shuffle, x_scale, w_scale, out, kernel_name
    )


def run_gemm_asm(x, weight_shuffle, x_scale, w_scale):
    from aiter.ops.gemm_op_a4w4 import gemm_mxfp4_asm

    return gemm_mxfp4_asm(
        x, weight_shuffle, x_scale, w_scale, dtype=dtypes.bf16, a_preshuffle=False
    )


def _compile_one(job):
    """Pool worker: compile one config into FlyDSL's disk cache. Returns (name, error)."""
    name, M, N, K = job
    try:
        from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
            preload_a4w4_kernel,
        )

        preload_a4w4_kernel(name, M, N, K)
        return name, None
    except Exception as e:  # noqa: BLE001
        return name, f"{type(e).__name__}: {str(e)[:200]}"


class GemmA4W4MxscaleBpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": TUNED_FILE,
        "untune_file": UNTUNED_FILE,
        # tight: a kernel that skips a few rows per tile differs on well under 5 %
        "errRatio": 0.001,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE",
    }

    def getKernelName(self, kernelId):
        from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a4w4_mxscale_bpreshuffle_wmma_common import (
            kernels_list,
        )

        ki = kernels_list.get(kernelId)
        return ki.name if ki is not None else None

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a4w4 as _op

        _op._a4w4_mxscale_config.cache_clear()
        _op._load_a4w4_mxscale_tuned.cache_clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--compile-workers",
            type=int,
            default=min(64, max(1, (os.cpu_count() or 2) // 2)),
            help="processes that compile candidate configs into the FlyDSL cache "
            "before they are timed (0: compile while timing, one at a time)",
        )
        self.parser.add_argument(
            "--confirm",
            type=int,
            default=4,
            help="fastest candidates per shape that the confirmation stage re-times",
        )
        self.parser.add_argument(
            "--confirm-rounds",
            type=int,
            default=3,
            help="interleaved rounds of the confirmation stage",
        )
        self.parser.add_argument(
            "--top-structures",
            type=int,
            default=12,
            help="general tile/warp/buffer structures per shape that stage 2 expands "
            "with every cluster shape and split-K",
        )

    def calculate(self, results, bpes=(0.5, 0.5, 2)):
        return super().calculate(results, bpes=bpes)

    # -- candidates ------------------------------------------------------------------
    def _precompile(self, jobs, workers):
        """Compile {name: (M, N, K)} into the disk cache; returns the names that failed."""
        if not jobs or workers < 1:
            return set()
        logger.info(f"compiling {len(jobs)} configs with {workers} workers")
        failed = set()
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx
        ) as pool:
            work = [(name, *shape) for name, shape in jobs.items()]
            for done, (name, err) in enumerate(pool.map(_compile_one, work), 1):
                if err:
                    failed.add(name)
                    logger.warning(f"compile failed {name}: {err}")
                if done % 100 == 0:
                    logger.info(f"  compiled {done}/{len(work)}")
        return failed

    def _build_tasks(
        self, per_shape, gfx, cu_num, warmup, iters, rounds=1, seed=0, with_asm=False
    ):
        """mp_tuner tasks of {(M, N, K): [(kernel_id, KernelInstance)]} (+ asm), interleaved
        across rounds."""
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_args = (["x_deq", "w_deq"], dtypes.bf16)
        run_kwargs = {"num_warmup": warmup, "num_iters": iters}
        tasks, tasks_data = [], []
        for (M, N, K), cands in per_shape.items():
            info_keys = (gfx, cu_num, M, N, K)
            runs = [
                (
                    (info_keys, i, ki.split_k, ki.name, "flydsl"),
                    run_gemm_flydsl,
                    (gemm_keys, ki.name),
                )
                for i, ki in cands
            ]
            if with_asm:
                runs.append(
                    (
                        (info_keys, 0, 0, asm_kernel_name(gfx, M, N, K), "asm"),
                        run_gemm_asm,
                        (gemm_keys[:4],),
                    )
                )
            for info, func, func_args in runs * rounds:
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        func,
                        func_args,
                        dict(run_kwargs),
                        run_torch,
                        ref_args,
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
            tasks_data.append((len(runs) * rounds, ()))
        return tasks, tasks_data

    def _time(self, per_shape, gfx, cu_num, args, iters, rounds=1, with_asm=False):
        """Compile then time the candidates; every result, no per-shape selection."""
        jobs = {}
        for shape, cands in per_shape.items():
            for _, ki in cands:
                jobs.setdefault(ki.name, shape)
        failed = self._precompile(jobs, args.compile_workers)
        if failed:
            per_shape = {
                shape: [(i, ki) for i, ki in cands if ki.name not in failed]
                for shape, cands in per_shape.items()
            }
        per_shape = {s: c for s, c in per_shape.items() if c or with_asm}
        tasks, tasks_data = self._build_tasks(
            per_shape, gfx, cu_num, args.warmup, iters, rounds, with_asm=with_asm
        )
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_data,
            args.mp,
            True,  # every result: stage 2 needs stage 1's ranking, post_process picks the best
            True,  # one process per shape: data and the fp32 reference are built once
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def tune(self, untunedf, tunedf, args):
        from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a4w4_mxscale_bpreshuffle_wmma_common import (
            candidates_for_shape,
            kernel_fits_shape,
            kernels_list,
        )

        gfx = self.get_gfx()
        if gfx != "gfx1250":
            logger.warning(f"gemm_a4w4 mxscale tuner covers gfx1250 only, got {gfx}")
            return []
        cu_num = self.get_cu_num()
        shapes = [
            (int(r.M), int(r.N), int(r.K)) for r in untunedf.itertuples(index=False)
        ]

        # stage 1: all compute candidates, general structures at a plain 1x1 / no split-K
        stage1 = {}
        for shape in shapes:
            cands = [
                (i, ki)
                for i, ki in candidates_for_shape(*shape)
                if ki.compute_bound
                or (ki.cluster_m, ki.cluster_n, ki.split_k) == (1, 1, 1)
            ]
            if not cands:
                logger.warning(
                    f"no a4w4 candidate fits M={shape[0]} N={shape[1]} K={shape[2]}"
                )
            stage1[shape] = cands
        logger.info(
            "stage 1: "
            + ", ".join(f"M={s[0]}:{len(c)}" for s, c in list(stage1.items())[:6])
            + (" ..." if len(stage1) > 6 else "")
        )
        screen_iters = max(20, args.iters // 3)
        results1 = self._time(stage1, gfx, cu_num, args, screen_iters)

        # stage 2: expand the fastest general structures with cluster shapes and split-K
        by_name = {ki.name: (i, ki) for i, ki in kernels_list.items()}
        structure = lambda ki: (
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
            ki.m_warp,
            ki.n_warp,
            ki.num_buffers,
        )
        best_by_shape = {}
        for info, us, err in results1:
            keys, _, _, name, _ = info
            ki = by_name[name][1]
            if ki.compute_bound or us <= 0 or err > args.errRatio:
                continue
            best_by_shape.setdefault(tuple(keys[2:]), []).append((us, ki))
        stage2 = {}
        for shape, ranked in best_by_shape.items():
            top = {
                structure(ki)
                for _, ki in sorted(ranked, key=lambda t: t[0])[: args.top_structures]
            }
            stage2[shape] = [
                (i, ki)
                for i, ki in kernels_list.items()
                if not ki.compute_bound
                and structure(ki) in top
                and (ki.cluster_m, ki.cluster_n, ki.split_k) != (1, 1, 1)
                and kernel_fits_shape(ki, *shape)
            ]
        results2 = (
            self._time(stage2, gfx, cu_num, args, screen_iters)
            if any(stage2.values())
            else []
        )

        # stage 3: the fastest of every shape and asm, full iterations, interleaved rounds
        ranked = {}
        for info, us, err in results1 + results2:
            if us > 0 and err <= args.errRatio:
                ranked.setdefault(tuple(info[0][2:]), []).append((us, info[3]))
        confirm = {
            shape: [
                by_name[name]
                for _, name in sorted(set(ranked.get(shape, ())))[: args.confirm]
            ]
            for shape in shapes
        }
        rounds = self._time(
            confirm, gfx, cu_num, args, args.iters, args.confirm_rounds, with_asm=True
        )
        grouped = {}
        for info, us, err in rounds:
            entry = grouped.setdefault((info[0], info[3]), [info, [], 0.0])
            entry[1].append(us)
            entry[2] = max(entry[2], err)
        results = []
        for info, times, err in grouped.values():
            valid = [t for t in times if t > 0]
            median = (
                round(statistics.median(valid), 4) if len(valid) == len(times) else -1
            )
            results.append((info, median, err))
        return results

    # -- results -----------------------------------------------------------------------
    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time in (self.INVALID_TIME, self.INF_TIME):
                kernelName = "None"
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "tflops": [tflops],
                    "bw": [bw],
                    "errRatio": [err_ratio],
                }
            )
            rows.append(key_dict)
        if not rows:
            return pd.DataFrame(columns=self.columns)
        return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)

    def run_config(self, args):
        """The production op on every shape: gemm_a4w4 with the tuned table in effect."""
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        for i in range(len(self.untunedf)):
            row = self.untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            shape_str = f"({M}, {N}, {K})"
            allowed, allowed_desc = self._get_run_config_err_ratio_limit(row, args)
            try:
                d = generate_data(M, N, K, 0)
                out, us = run_perftest(
                    gemm_a4w4,
                    d["x"],
                    d["weight_shuffle"],
                    d["x_scale"],
                    d["w_scale"],
                    dtype=dtypes.bf16,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                err_ratio = checkAllclose(
                    out,
                    run_torch(d["x_deq"], d["w_deq"]),
                    msg=f"run_config {shape_str}",
                )
                status = (
                    "ok"
                    if err_ratio <= allowed
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmA4W4MxscaleBpreshuffleTuner(
        "a4w4_mxscale_bpreshuffle_tuned_gemm",
        key=key,
        resultList=resultList,
        description="Tune gemm_a4w4 on gfx1250 (FlyDSL compute + general kernels, "
        "MXFP4 with 1x32 e8m0 scales in the 32x4 layout)",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
