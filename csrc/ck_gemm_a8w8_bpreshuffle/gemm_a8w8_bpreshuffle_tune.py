# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import sys
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F
from gemm_a8w8_bpreshuffle_common import kernels_list as kernels_list_ck

import aiter
from aiter import dtypes
from aiter.jit.core import (
    AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
    AITER_CSRC_DIR,
    get_asm_dir,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

sys.path.insert(0, f"{AITER_CSRC_DIR}/cktile_gemm_a8w8_bpreshuffle/")
from gemm_a8w8_bpreshuffle_cktile_common import (
    BLOCK_PER_CU_MAX,
)
from gemm_a8w8_bpreshuffle_cktile_common import (
    kernels_list as kernels_list_cktile,
)

# Both a8w8 bpreshuffle pipelines live in one common and are enumerated through
# FLYDSL_PIPELINES. The guard that kept a broken 8wave table from disabling
# preshuffle moved into that module, around the 8wave op-module import; the
# per-table aliases below are only for the runners and get_kernel_name.
try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        PIPELINES as FLYDSL_PIPELINES,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        kernels_list as kernels_list_flydsl,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        kernels_list_8wave as kernels_list_flydsl_8wave,
    )
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a8w8_bpreshuffle_common.py not found, flydsl tuning disabled"
    )
    FLYDSL_PIPELINES = ()
    kernels_list_flydsl = {}
    kernels_list_flydsl_8wave = {}


from aiter.ops.flydsl.utils import is_flydsl_available

if is_flydsl_available():
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8


def get_valid_asm_splitK_list(K: int, max_splitK: int, tile_k: int = 128):
    """Filter splitK values to only those that produce valid TileK-aligned partitions."""
    valid = []
    for sk in range(1, max_splitK + 1):
        k_per_split = (K + sk - 1) // sk
        k_per_split_aligned = ((k_per_split + tile_k - 1) // tile_k) * tile_k
        actual_ksplit = (K + k_per_split_aligned - 1) // k_per_split_aligned
        if actual_ksplit == sk:
            valid.append(sk)
    return valid if valid else [1]


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        return not percent > 0.01


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, out, kernel_id, splitK=0):
    aiter.gemm_a8w8_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_bpreshuffle_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK=0
):
    aiter.gemm_a8w8_bpreshuffle_cktile_tune(
        x, weight, x_scale, w_scale, out, kernel_id, splitK
    )
    return out


def run_gemm_a8w8_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    bias,
    kernelName,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    splitK=None,
):

    return aiter.gemm_a8w8_asm(
        x,
        weight,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        splitK=splitK,
    )


def run_gemm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_id):
    ki = kernels_list_flydsl[kernel_id]
    flydsl_preshuffle_gemm_a8(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        ki.use_async_copy,
        ki.waves_per_eu,
        ki.xcd_swizzle,
        ki.lds_stage,
        ki.enable_scheduler,
    )
    return out


def run_gemm_flydsl_8wave(x, weight_shuffle, x_scale, w_scale, out, kernel_id):
    from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import flydsl_8wave_gemm_a8

    ki = kernels_list_flydsl_8wave[kernel_id]
    flydsl_8wave_gemm_a8(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.block_m,
        ki.block_n,
        waves_per_eu=ki.waves_per_eu,
        xcd_swizzle=ki.xcd_swizzle,
    )
    return out


# Pipeline name -> (runner, is-it-usable-right-now). The availability probe is a
# callable, and deliberately different per pipeline: preshuffle needs the symbol
# the is_flydsl_available() gate above binds at import, while 8wave imports its
# entry point lazily inside its runner.
_FLYDSL_PIPELINE_RUNNERS = {
    "preshuffle": (
        run_gemm_flydsl,
        lambda: "flydsl_preshuffle_gemm_a8" in globals(),
    ),
    "8wave": (run_gemm_flydsl_8wave, is_flydsl_available),
}

# The tuner speaks torch dtypes, Pipeline.q_dtypes_w speaks names (that module
# must stay importable without torch).
_Q_DTYPE_W_NAMES = {dtypes.fp8: "fp8", dtypes.i8: "int8"}


def run_gemm_flydsl_gfx1250(x, weight_shuffle, x_scale, w_scale, out, kernel_id):
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        run_preshuffle_gemm_a8_gfx1250,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
        kernels_list as kernels_list_flydsl_wmma,
    )

    ki = kernels_list_flydsl_wmma[kernel_id]
    run_preshuffle_gemm_a8_gfx1250(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        num_buffers=ki.num_buffers,
        split_k=ki.split_k,
        cluster_m=ki.cluster_m,
        cluster_n=ki.cluster_n,
        m_warp=ki.m_warp,
        n_warp=ki.n_warp,
    )
    return out


def generate_data(
    m, n, k, seed, dtype=dtypes.bf16, q_dtype_w=dtypes.fp8, is_asm=False, device="cuda"
):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, device=device)
    weight = torch.randn((n, k), dtype=dtype, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=q_dtype_w)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=q_dtype_w)
    bias_f32 = None
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    if is_asm:
        pad_k = 128
        x_full = torch.empty_strided(
            (m, k + pad_k),
            (k + pad_k, 1),
            dtype=x.dtype,
            device=x.device,
        )
        x_full[:, :k] = x
        x = x_full[:, :k]
        bias = torch.zeros(1, n, dtype=dtype, device=device)
        bias_f32 = bias.to(dtypes.fp32)
    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "x": x,
        "weight_shuffle": weight_shuffle,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
        "weight": weight,
        "bias_f32": bias_f32,
    }


def libtype_list(string):
    values = string.split(",")
    for value in values:
        if value not in ["all", "asm", "ck", "cktile", "flydsl"]:
            raise argparse.ArgumentTypeError(f"Invalid libtype: {value}")
    return values


class GemmA8W8BpreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_GEMM_config_with_quant_type.cache_clear()
        _op._GEMM_QUANT_TYPE_CACHE.clear()
        _op._GEMM_QUANT_TYPE_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--libtype",
            # nargs='+',
            # choices=['all', 'asm', 'ck', 'cktile'],
            type=libtype_list,
            default=["all"],
            required=False,
            help="choose libtype to be tuned, support ['all', 'asm', 'ck', 'cktile', 'flydsl']",
        )

        self.parser.add_argument(
            "--blockPerCu",
            nargs="+",
            type=int,
            default=list(range(1, BLOCK_PER_CU_MAX + 1)),
            help="List of BlockPerCu values to tune (CKTile only)",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="ck"):
        if libtype == "ck":
            if kernelId < 0 or kernelId > len(kernels_list_ck):
                return None
            kernelList = kernels_list_ck
        elif libtype == "cktile":
            if kernelId < 0 or kernelId > len(kernels_list_cktile):
                return None
            kernelList = kernels_list_cktile
        elif libtype == "flydsl":
            # Several candidate tables share this libtype; their id ranges are
            # disjoint (see KERNEL_ID_BASE_8WAVE), so ask each in turn.
            for pipe in FLYDSL_PIPELINES:
                if kernelId in pipe.kernels_list:
                    return pipe.kernels_list[kernelId].name
            return None
        else:
            return None
        return kernelList[kernelId].name

    def get_asm_kernels(self, file):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        shuffle_df = (
            df[df["bpreshuffle"] == 1]
            .reset_index()
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            shuffle_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_asm_gemm_i8_tasks(self, info_keys, useSplitK, kernel_id_start, seed=0):
        task = []
        _gfx, _cu_num, M, N, K, q_dtype_w = info_keys
        if eval(q_dtype_w) != dtypes.i8:
            return task
        asm_kernel_list_csv = f"{get_asm_dir()}/i8gemm/i8gemm_bf16_perTokenI8.csv"
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv)
        asm_tiles = [key for key in asm_kernels]

        gemm_asm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out", "bias_f32"]
        ref_keys = ["x", "weight", "x_scale", "w_scale", "bias_f32"]
        asm_kernels_id = kernel_id_start
        for key in asm_tiles:
            tile_m, tile_n, splitk = key
            kernelName = asm_kernels.get((tile_m, tile_n, splitk), [])
            if len(kernelName) == 0:
                print(f"no kernel name for ({tile_m}, {tile_n})!!!!")
                continue
            if useSplitK and splitk != 0:
                splitK_list = get_valid_asm_splitK_list(K, 8)
            else:
                splitK_list = [1]
            for splitK in splitK_list:
                kernel_name = kernelName[0]
                info = (info_keys, asm_kernels_id, splitK, kernel_name, "asm")
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w), True),
                        run_gemm_a8w8_asm,
                        (
                            gemm_asm_keys,
                            kernel_name,
                            dtypes.bf16,
                            True,
                            splitK,
                        ),
                        {
                            "num_warmup": 10,
                            "num_iters": 101,
                        },
                        run_torch,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                        None,
                    )
                )
            asm_kernels_id = asm_kernels_id + 1
        return task

    def get_cktile_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        _gfx, _cu_num, M, N, K, q_dtype_w = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"Warning: q_dtype_w only support {dtypes.fp8}, actual q_dtype_w is {q_dtype_w}!"
            )
            return []
        filtered_cktile = {
            k: v
            for k, v in kernels_list_cktile.items()
            if v.BlockPerCu in args.blockPerCu
        }
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale", "bias_f32"]
        tasks_ck = []
        for i, kernel in filtered_cktile.items():
            maxsplitK = (
                aiter.compute_gemm_SplitK(
                    M,
                    N,
                    K,
                    kernel.MTile,
                    kernel.NTile,
                    kernel.KTile,
                )
                if useSplitK
                else 0
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "cktile")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_bpreshuffle_cktile,
                        (
                            gemm_keys,
                            i,
                            splitK,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_ck

    def get_ck_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
    ):
        _gfx, _cu_num, M, N, K, q_dtype_w = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"Warning: q_dtype_w only support {dtypes.fp8}, actual q_dtype_w is {q_dtype_w}!"
            )
            return []
        kernels_num = len(kernels_list_ck)
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale", "bias_f32"]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernels_list_ck[i]
            maxsplitK = (
                aiter.compute_gemm_SplitK(
                    M,
                    N,
                    K,
                    kernel.MPerBLOCK,
                    kernel.NPerBLOCK,
                    kernel.KPerBLOCK,
                )
                if useSplitK
                else 0
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "ck")
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, eval(q_dtype_w)),
                        run_gemm_a8w8_bpreshuffle,
                        (
                            gemm_keys,
                            i,
                            splitK,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_ck

    def get_flydsl_gemm_a8w8_bpreshuffle_tune_task(
        self,
        info_keys,
        seed,
    ):
        gfx, _cu_num, M, N, K, q_dtype_w = info_keys

        if gfx == "gfx1250":
            return self._get_flydsl_tune_task_gfx1250(info_keys, seed)

        q_dtype_eval = eval(q_dtype_w)
        q_dtype_name = _Q_DTYPE_W_NAMES.get(q_dtype_eval)
        if q_dtype_name is None:
            print(f"[FlyDSL] unsupported q_dtype_w {q_dtype_w}, skipping")
            return []

        gemm_flydsl_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale", "bias_f32"]
        tasks = []
        # One pass per pipeline, in PIPELINES order (preshuffle, then 8wave).
        # Sweeping them under one libtype is what lets a single
        # ``--libtype flydsl`` run compare them and pick one winner per shape;
        # only the candidate table, the predicate and the runner differ.
        for pipe in FLYDSL_PIPELINES:
            runner_entry = _FLYDSL_PIPELINE_RUNNERS.get(pipe.name)
            if runner_entry is None:
                print(f"[FlyDSL] no runner registered for pipeline {pipe.name!r}")
                continue
            runner, is_available = runner_entry
            if q_dtype_name not in pipe.q_dtypes_w:
                continue
            if not pipe.kernels_list or not is_available():
                continue
            for i in sorted(pipe.kernels_list.keys()):
                ki = pipe.kernels_list[i]
                if not pipe.fits(ki, M, N, K):
                    continue
                tasks.append(
                    (
                        (info_keys, i, 0, ki.name, "flydsl"),
                        generate_data,
                        (M, N, K, seed, dtypes.bf16, q_dtype_eval),
                        runner,
                        (
                            gemm_flydsl_keys,
                            i,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks

    def _get_flydsl_tune_task_gfx1250(self, info_keys, seed):
        """gfx1250 WMMA ptpc tuning tasks for the FlyDSL libtype."""
        _gfx, _cu_num, M, N, K, q_dtype_w = info_keys
        if eval(q_dtype_w) != dtypes.fp8:
            print(
                f"[FlyDSL][gfx1250] WMMA ptpc supports fp8 only, skipping {q_dtype_w}"
            )
            return []
        if not is_flydsl_available():
            return []
        try:
            from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
                kernel_fits_shape as kernel_fits_shape_wmma,
            )
            from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
                kernels_list as kernels_list_flydsl_wmma,
            )
        except ImportError:
            return []
        if not kernels_list_flydsl_wmma:
            return []
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale", "bias_f32"]
        tasks = []
        for i in sorted(kernels_list_flydsl_wmma.keys()):
            ki = kernels_list_flydsl_wmma[i]
            if not kernel_fits_shape_wmma(ki, M, N, K):
                continue
            info = (info_keys, i, 0, ki.name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, dtypes.bf16, dtypes.fp8),
                    run_gemm_flydsl_gfx1250,
                    (
                        gemm_keys,
                        i,
                    ),
                    {
                        "num_warmup": args.warmup,
                        "num_iters": args.iters,
                    },
                    run_torch,
                    (
                        ref_keys,
                        dtypes.bf16,
                    ),
                    {},
                    None,
                    1e-2,
                    0.01,
                    None,
                    None,
                    ("out",),
                )
            )
        return tasks

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 0
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            q_dtype_w = untunedf.loc[i, "q_dtype_w"]
            seed = seed + 1
            prev_task_count = len(task)
            info_keys = (gfx, cu_num, M, N, K, q_dtype_w)
            if "all" in args.libtype or "ck" in args.libtype:
                task.extend(
                    self.get_ck_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                    )
                )
            if "all" in args.libtype or "cktile" in args.libtype:
                task.extend(
                    self.get_cktile_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                    )
                )
            if "all" in args.libtype or "asm" in args.libtype:
                task.extend(self.get_asm_gemm_i8_tasks(info_keys, useSplitK, 0, seed))
            if "all" in args.libtype or "flydsl" in args.libtype:
                task.extend(
                    self.get_flydsl_gemm_a8w8_bpreshuffle_tune_task(
                        info_keys,
                        seed,
                    )
                )

            shape_kernel_nums = len(task) - prev_task_count

            tasks_data.append((shape_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        return ret

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (
                    self.getKernelName(kernelId, libtype)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} {kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            if resultdf.empty:
                resultdf = temp
            else:
                resultdf = pd.concat([resultdf, temp], ignore_index=True)
        return resultdf

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_ASM, gemm_a8w8_bpreshuffle
        from aiter.test_common import checkAllclose, run_perftest

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            q_dtype_w = row["q_dtype_w"]
            shape_str = f"({M}, {N}, {K}, {q_dtype_w})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                is_asm = eval(q_dtype_w) == dtypes.i8
                gd = generate_data(
                    M,
                    N,
                    K,
                    0,
                    dtypes.bf16,
                    eval(q_dtype_w),
                    is_asm,
                )
                x = gd["x"]
                weight_shuffle = gd["weight_shuffle"]
                x_scale = gd["x_scale"]
                w_scale = gd["w_scale"]
                out = gd["out"]
                weight = gd["weight"]
                bias_f32 = gd["bias_f32"]
                if is_asm:
                    out, us = run_perftest(
                        gemm_a8w8_ASM,
                        x,
                        weight_shuffle,
                        x_scale,
                        w_scale,
                        bias_f32,
                        num_warmup=args.warmup,
                        num_iters=args.iters,
                    )
                else:
                    out, us = run_perftest(
                        gemm_a8w8_bpreshuffle,
                        x,
                        weight_shuffle,
                        x_scale,
                        w_scale,
                        num_warmup=args.warmup,
                        num_iters=args.iters,
                    )
                ref = run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
                err_ratio = checkAllclose(
                    out.to(dtypes.bf16), ref, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
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
    ## use default key and resultList
    key = ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
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
    tuner = GemmA8W8BpreShuffleTuner(
        "GemmA8W8BpreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for gemm a8w8 bpreshuffle kernel",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
