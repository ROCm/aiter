# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Multi-backend bf16 / a16w16 GEMM tuner.

Follows the csrc tuner pattern (like ck_gemm_a8w8, ck_gemm_a8w8_blockscale).
Backends: asm, opus, flydsl, triton, skinny, torch.
hipblaslt is opt-in via --with-hipblaslt (imports from gradlib).
"""

import argparse
import functools
import os
import sys
from functools import lru_cache

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes, logger
from aiter.jit.core import AITER_CONFIG_GEMM_BF16, get_asm_dir
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.gemm_op_a16w16 import ASM_SPLITK_MAX_GRID
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------

FLYDSL_TUNE_ERROR = None
try:
    if is_flydsl_available():
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_hgemm,
            get_flydsl_splitk_hgemm_kernels,
        )
    else:
        raise ImportError("flydsl package is not installed")
except ImportError as exc:
    flydsl_hgemm = None
    get_flydsl_splitk_hgemm_kernels = None
    FLYDSL_TUNE_ERROR = str(exc)

OPUS_TUNE_ERROR = None
try:
    _opus_csrc = os.path.join(os.path.dirname(__file__), "../opus_gemm")
    if _opus_csrc not in sys.path:
        sys.path.insert(0, os.path.abspath(_opus_csrc))
    from opus_gemm_common import kernels_list as _opus_kernels_list
    from opus_gemm_tune import (
        candidate_kids_for_shape as _opus_candidate_kids_for_shape,
        candidate_splitK as _opus_candidate_splitK,
        kid_rejects_shape as _opus_kid_rejects_shape,
        kid_rejects_bias as _opus_kid_rejects_bias,
        _ensure_kids_compiled as _opus_ensure_kids_compiled,
    )
    from aiter.ops.opus.gemm_op_a16w16 import (
        opus_gemm_a16w16_tune as _opus_gemm_a16w16_tune,
    )

    _opus_all_kernels = dict(_opus_kernels_list)
except Exception as _opus_exc:
    _opus_gemm_a16w16_tune = None
    _opus_all_kernels = None
    _opus_candidate_kids_for_shape = None
    _opus_kid_rejects_shape = None
    _opus_kid_rejects_bias = None
    _opus_candidate_splitK = None
    _opus_ensure_kids_compiled = None
    OPUS_TUNE_ERROR = str(_opus_exc)

HIPBLASLT_TUNE_ERROR = None
try:
    _gradlib_path = os.path.join(os.path.dirname(__file__), "../../gradlib/gradlib")
    if _gradlib_path not in sys.path:
        sys.path.insert(0, os.path.abspath(_gradlib_path))
    from GemmTuner import Gemm as HipblasltGemm
except Exception as _hipb_exc:
    HipblasltGemm = None
    HIPBLASLT_TUNE_ERROR = str(_hipb_exc)

# ---------------------------------------------------------------------------
# Data generation & reference
# ---------------------------------------------------------------------------


def generate_data(
    m,
    n,
    k,
    indtype,
    outdtype,
    scaleAB,
    is_shuffle=False,
    seed=0,
    bias=False,
    device="cuda:0",
):
    torch.manual_seed(seed)
    if indtype == dtypes.fp8:
        randn_dtype = dtypes.bf16
    else:
        randn_dtype = indtype
    inp = torch.randn((m, k), device=device).to(randn_dtype)
    weights = torch.randn((n, k), device=device).to(randn_dtype)
    if indtype == dtypes.fp8:
        inp, x_scale = aiter.pertoken_quant(inp, quant_dtype=dtypes.fp8)
        weights, w_scale = aiter.pertoken_quant(weights, quant_dtype=dtypes.fp8)
    else:
        scale_half = torch.tensor(0.5, dtype=dtypes.fp32, device=device)
        w_scale = scale_half
        x_scale = scale_half
    if is_shuffle:
        from aiter.ops.shuffle import shuffle_weight

        shuffleweights = shuffle_weight(weights, layout=(16, 16))
    else:
        shuffleweights = weights

    bias = torch.randn(n, device=device).to(outdtype) if bias else None
    out_asm = torch.empty(m, n, dtype=outdtype, device=device)
    return {
        "inp": inp,
        "weights": weights,
        "weights_t": weights.t(),
        "bias": bias,
        "x_scale": x_scale,
        "out_asm": out_asm,
        "shuffleweights": shuffleweights,
        "w_scale": w_scale,
    }


def get_gemm_ref(inp, weights, bias, scaleA, scaleB, indtype, outdtype):
    if indtype == dtypes.fp8:
        x = inp.to(dtypes.fp32) * scaleA
        weight = weights.to(dtypes.fp32) * scaleB
        out = F.linear(x, weight)
        if bias is not None:
            out = out.to(bias) + bias
        return out.to(outdtype)
    else:
        ref = (
            (
                F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32))
                + bias.to(dtypes.fp32)
            ).to(outdtype)
            if bias is not None
            else F.linear(inp.to(dtypes.fp32), weights.to(dtypes.fp32)).to(outdtype)
        )
    return ref


# ---------------------------------------------------------------------------
# Per-backend run functions
# ---------------------------------------------------------------------------


def run_gemm_bf16_asm(
    inp, w, out, bias=None, splitK=None, kernelName=None, bpreshuffle=False
):
    return aiter.gemm_a16w16_asm(
        inp,
        w,
        out,
        bias=bias,
        splitK=splitK,
        kernelName=kernelName,
        bpreshuffle=bpreshuffle,
    )


def run_triton_gemm_bf16(input, weight, bias=None, otype=dtypes.bf16):
    return triton_gemm_a16w16(input, weight, bias=bias, dtype=otype)


_opus_max_delta_checked = set()


def run_opus_gemm_bf16(inp, weight, out, bias=None, kid=0, splitK=0):
    inp3 = inp.unsqueeze(0)
    weight3 = weight.unsqueeze(0)
    out3 = out.unsqueeze(0)
    _opus_gemm_a16w16_tune(
        inp3,
        weight3,
        out3,
        bias=bias,
        kernelId=kid,
        splitK=splitK,
    )
    if torch.cuda.is_current_stream_capturing():
        return out
    cache_key = (
        kid,
        splitK,
        inp.size(0),
        weight.size(0),
        inp.size(-1),
        bias is not None,
        str(out.dtype),
    )
    if cache_key in _opus_max_delta_checked:
        return out
    ref_fp32 = torch.bmm(inp3.float(), weight3.float().transpose(-1, -2))
    if bias is not None:
        if bias.dim() == 1:
            ref_fp32 = ref_fp32 + bias.float().view(1, 1, -1)
        else:
            ref_fp32 = ref_fp32 + bias.float().unsqueeze(1)
    max_delta = (out3.float() - ref_fp32).abs().max().item()
    max_ref = ref_fp32.abs().max().item()
    bound = max(max_ref * 0.1, 1.0)
    if max_delta > bound:
        raise RuntimeError(
            f"opus maxDelta {max_delta:.3f} > bound {bound:.3f} "
            f"(max|ref|={max_ref:.3f}, scale=0.1) "
            f"for kid={kid} splitK={splitK} bias={bias is not None} "
            f"M={inp.size(0)} N={weight.size(0)} K={inp.size(-1)}"
        )
    _opus_max_delta_checked.add(cache_key)
    return out


@lru_cache(maxsize=1)
def get_native_gemm_funcs():
    from aiter.tuned_gemm import is_skinny_default_shape, skinny_gemm, torch_gemm

    return torch_gemm, skinny_gemm, is_skinny_default_shape


def run_torch_gemm_a16w16(
    input,
    weight,
    bias=None,
    scale_a=None,
    scale_b=None,
    otype=dtypes.bf16,
):
    native_torch_gemm, _, _ = get_native_gemm_funcs()
    return native_torch_gemm(
        input,
        weight,
        0,
        bias=bias,
        otype=otype,
        scale_a=scale_a,
        scale_b=scale_b,
    )


def run_skinny_gemm_a16w16(input, weight, bias=None, otype=dtypes.bf16):
    _, native_skinny_gemm, _ = get_native_gemm_funcs()
    return native_skinny_gemm(input, weight, 2, bias=bias, otype=otype)


def run_flydsl_gemm_bf16(input, weight, bias=None, otype=dtypes.bf16, config=None):
    if flydsl_hgemm is None:
        raise RuntimeError(f"flydsl is not available for tuning: {FLYDSL_TUNE_ERROR}")
    if config is None:
        raise ValueError("flydsl tuning requires a kernel config")
    stages = config.get("stages", config.get("stage", 2))
    fused_bias = None
    if (
        bias is not None
        and (otype is None or otype == input.dtype)
        and bias.dtype == input.dtype
    ):
        fused_bias = bias
    out = flydsl_hgemm(
        input,
        weight,
        bias=fused_bias,
        kernel_family=config.get("kernel_family"),
        tile_m=config["tile_m"],
        tile_n=config["tile_n"],
        tile_k=config["tile_k"],
        split_k=config["split_k"],
        block_m_warps=config["block_m_warps"],
        block_n_warps=config["block_n_warps"],
        block_k_warps=config["block_k_warps"],
        n_tile_repeat=config.get("n_tile_repeat", 1),
        persistent_n_tiles=config.get("persistent_n_tiles", 1),
        waves_per_eu=config.get("waves_per_eu", 0),
        b_to_lds_unroll=config.get("b_to_lds_unroll", 0),
        stages=stages,
        async_copy=config.get("async_copy", False),
        b_to_lds=config["b_to_lds"],
        b_preshuffle=config.get("b_preshuffle", False),
        auto_shuffle_b=False,
        c_to_lds=config.get("c_to_lds", False),
    )
    if bias is not None and fused_bias is None:
        out = out.to(bias.dtype) + bias
    if otype is not None and out.dtype != otype:
        out = out.to(otype)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_flydsl_bf16_catalog(m: int, n: int, k: int):
    if get_flydsl_splitk_hgemm_kernels is None:
        return []
    kernels = get_flydsl_splitk_hgemm_kernels("bf16", "bf16", m=m, n=n, k=k)
    catalog = [
        (idx, name, dict(kernels[name])) for idx, name in enumerate(sorted(kernels))
    ]
    logger.info(
        f"FlyDSL bf16 catalog size for M={m}, N={n}, K={k}: {len(catalog)} kernels"
    )
    return catalog


@functools.lru_cache(maxsize=1024)
def compute_gemm_SplitK(M, N, K, tile_m, tile_n, tile_k):
    cu_num = get_cu_num()
    tile_num = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    if tile_num < cu_num:
        return int(cu_num / tile_num)
    return 4


def get_asm_kernels(file, is_shuffle=False):
    if not os.path.exists(file):
        print(f"ASM kernel list file not exist: {file}")
        return {}
    df = pd.read_csv(file)
    return (
        df.groupby(["tileM", "tileN", "pf", "splitK", "subK", "bias", "bPreshuffle"])[
            "knl_name"
        ]
        .apply(list)
        .to_dict()
    )


ALL_LIBTYPES = [
    "all",
    "asm",
    "hipblaslt",
    "triton",
    "flydsl",
    "torch",
    "skinny",
    "opus",
]


def libtype_list(string):
    values = string.split(",")
    for value in values:
        if value not in ALL_LIBTYPES:
            raise argparse.ArgumentTypeError(f"Invalid libtype: {value}")
    return values


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------


class GemmA16W16Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_BF16}",
        "untune_file": "aiter/configs/bf16_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_BF16",
    }

    def __init__(self, name, keys, resultList, description=""):
        super().__init__(name, keys, resultList, description)

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--tuned_file",
            type=str,
            default=os.getenv("GTUNE_TUNED", AITER_CONFIG_GEMM_BF16),
            dest="tune_file",
            help="output file for tuned gemm solutions",
        )
        self.parser.add_argument(
            "--input_file",
            type=str,
            default=os.getenv("GTUNE_INPUT", None),
            dest="untune_file",
            help="list of gemms to tune for, mutually exclusive with model_dir",
        )
        self.parser.add_argument(
            "--indtype",
            type=str,
            default=None,
            choices=["f32", "f16", "bf16", "fp8"],
            help="dtype override for all shapes",
        )
        self.parser.add_argument(
            "--outdtype",
            type=str,
            default=None,
            choices=["f32", "f16", "bf16", "fp8"],
            help="output dtype override",
        )
        self.parser.add_argument(
            "--all_bias",
            action="store_true",
            help="Tune for both bias and non-bias cases",
        )
        self.parser.add_argument(
            "--libtype",
            type=libtype_list,
            default=["all"],
            required=False,
            help="choose libtype to tune: all, asm, hipblaslt, triton, flydsl, torch, skinny, opus. "
            "hipblaslt requires --with-hipblaslt.",
        )
        self.parser.add_argument(
            "--with-hipblaslt",
            action="store_true",
            default=False,
            dest="with_hipblaslt",
            help="Include hipblaslt in tuning (disabled by default). "
            "hipblaslt tuning is also available standalone via gradlib/gradlib/gemm_tuner.py.",
        )

    def _clear_op_caches(self):
        from aiter.tuned_gemm import get_GEMM_A16W16_config_, get_GEMM_A16W16_config

        get_GEMM_A16W16_config_.cache_clear()
        get_GEMM_A16W16_config.cache_clear()

    def getKernelName(self, kernelId):
        return None

    def calculate(self, results, bpes=(2, 2, 2)):
        return super().calculate(results, bpes=(2, 2, 2))

    def run_config(self, args):
        from aiter.tuned_gemm import gemm_a16w16
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            bias = row["bias"]
            indtype = str(row["dtype"])
            outdtype = str(row["outdtype"])
            scaleAB = row["scaleAB"]
            bpreshuffle = row["bpreshuffle"]
            shape_str = f"({M}, {N}, {K}, {indtype}, bias={bias})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                data = generate_data(
                    M,
                    N,
                    K,
                    eval(indtype),
                    eval(outdtype),
                    scaleAB,
                    bpreshuffle,
                    0,
                    bias,
                )
                inp = data["inp"]
                weights = data["weights"]
                bias_tensor = data["bias"]
                x_scale, w_scale = data["x_scale"], data["w_scale"]
                shuffleweights = data["shuffleweights"]
                w = shuffleweights if bpreshuffle else weights
                scale_a = x_scale if scaleAB else None
                scale_b = w_scale if scaleAB else None
                out, us = run_perftest(
                    gemm_a16w16,
                    inp,
                    w,
                    bias=bias_tensor,
                    otype=eval(outdtype),
                    scale_a=scale_a,
                    scale_b=scale_b,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = get_gemm_ref(
                    inp,
                    weights,
                    bias_tensor,
                    x_scale,
                    w_scale,
                    eval(indtype),
                    eval(outdtype),
                )
                _atol = 5e-2 if eval(outdtype) == torch.bfloat16 else 1e-2
                _rtol = _atol
                err_ratio = checkAllclose(
                    out, ref, atol=_atol, rtol=_rtol, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def get_untuned_gemm_list(self, untuned_gemm_file):
        assert os.path.exists(
            untuned_gemm_file
        ), f"Not exist untuned file: {untuned_gemm_file}"
        untunedf = pd.read_csv(untuned_gemm_file).fillna("")
        return untunedf.drop_duplicates().reset_index(drop=True)

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)
            if "outdtype" not in self.untunedf.columns:
                self.untunedf["outdtype"] = self.untunedf["dtype"]
            if "scaleAB" not in self.untunedf.columns:
                self.untunedf["scaleAB"] = False
            _cli_to_dtypes = {
                "f16": "fp16",
                "f32": "fp32",
                "bf16": "bf16",
                "fp8": "fp8",
            }
            if args.indtype is not None:
                self.untunedf["dtype"] = f"dtypes.{_cli_to_dtypes[args.indtype]}"
            if args.outdtype is not None:
                self.untunedf["outdtype"] = f"dtypes.{_cli_to_dtypes[args.outdtype]}"
            self.tunedf = self.get_tuned_gemm_list(self.get_out_file(args.tune_file))
            self.untunedf["gfx"] = self.get_gfx()
            self.untunedf["cu_num"] = self.get_cu_num()
            self.untunedf = self.untunedf[self.keys]
            untunedf_cols = self.untunedf.columns
            if len(self.tunedf) != 0:
                mask = self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[untunedf_cols].apply(tuple, axis=1)
                )
                if args.verbose:
                    logger.info("skiped tuned shapes:")
                    print(self.untunedf[mask])
                self.untunedf = self.untunedf[~mask]
            self.untunedf = self.untunedf.drop_duplicates().reset_index(drop=True)
            print("untunedf is ", self.untunedf)

    # -------------------------------------------------------------------
    # Per-backend task builders (called from tune())
    # -------------------------------------------------------------------

    def _get_asm_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        if (scaleAB or K % 64 != 0 or indtype != dtypes.bf16) and get_gfx() == "gfx942":
            return []
        if (
            scaleAB or K % 64 != 0 or N % 64 != 0 or indtype != dtypes.bf16
        ) and get_gfx() == "gfx950":
            return []
        asm_kernel_list_csv = f"{get_asm_dir()}/bf16gemm/bf16gemm_fp32bf16.csv"
        asm_kernels = get_asm_kernels(asm_kernel_list_csv, is_shuffle)
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        atol = rtol
        tasks = []
        solidx = 0
        for key in asm_kernels.keys():
            tile_m, tile_n, pf, splitK_flag, subK, bias_flag, bPreshuffle = key
            kernelName = asm_kernels[key][0]
            start = 1
            if splitK_flag:
                maxSplitK = compute_gemm_SplitK(M, N, K, tile_m, tile_n, 256)
                start = 2
            else:
                maxSplitK = 1
            maxSplitK = min(maxSplitK, 16)
            if not bias_flag and has_bias:
                continue
            if (bPreshuffle == 0 and is_shuffle) or (
                bPreshuffle == 1 and not is_shuffle
            ):
                continue
            solidx += 1
            for sk in range(start, maxSplitK + 1):
                if K / sk < subK:
                    break
                if sk > 1:
                    gdx = (N + tile_n - 1) // tile_n
                    gdy = (M + tile_m - 1) // tile_m
                    if gdx * gdy > ASM_SPLITK_MAX_GRID:
                        continue
                info = (info_keys, solidx, sk, kernelName, "asm", is_shuffle)
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                        run_gemm_bf16_asm,
                        (
                            ["inp", "shuffleweights", "out_asm", "bias"],
                            sk,
                            kernelName,
                            is_shuffle,
                        ),
                        dict(run_kwargs),
                        get_gemm_ref,
                        (
                            ["inp", "weights", "bias", "x_scale", "w_scale"],
                            indtype,
                            outdtype,
                        ),
                        {},
                        None,
                        rtol,
                        atol,
                        None,
                        None,
                        ("out_asm",),
                    )
                )
        return tasks

    def _get_opus_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        if _opus_gemm_a16w16_tune is None:
            logger.warning(f"opus not available, skip. reason: {OPUS_TUNE_ERROR}")
            return []
        if scaleAB or indtype != dtypes.bf16:
            return []
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        cu_num = get_cu_num()
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        atol = rtol
        cand_kids = _opus_candidate_kids_for_shape(M, N, K, has_bias, cu_num)
        tasks = []
        for kid in sorted(cand_kids):
            k_inst = _opus_all_kernels.get(kid)
            if k_inst is None:
                continue
            if _opus_kid_rejects_shape(k_inst, M, N, K):
                continue
            if _opus_kid_rejects_bias(k_inst, has_bias):
                continue
            if k_inst.kernel_tag == "a16w16_flatmm_splitk":
                splitK_range = _opus_candidate_splitK(M, N, K, 1, cu_num, k_inst)
            else:
                splitK_range = [0]
            for sk in splitK_range:
                info = (info_keys, kid, sk, k_inst.name, "opus", is_shuffle)
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                        run_opus_gemm_bf16,
                        (["inp", "weights", "out_asm", "bias"], kid, sk),
                        dict(run_kwargs),
                        get_gemm_ref,
                        (
                            ["inp", "weights", "bias", "x_scale", "w_scale"],
                            indtype,
                            outdtype,
                        ),
                        {},
                        None,
                        rtol,
                        atol,
                        None,
                        None,
                        ("out_asm",),
                    )
                )
        logger.info(f"opus candidate count for M={M}, N={N}, K={K}: {len(tasks)}")
        return tasks

    def _get_flydsl_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        if flydsl_hgemm is None or get_flydsl_splitk_hgemm_kernels is None:
            logger.warning(f"FlyDSL not available, skip. reason: {FLYDSL_TUNE_ERROR}")
            return []
        if scaleAB or indtype != dtypes.bf16:
            return []
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        atol = rtol
        flydsl_catalog = get_flydsl_bf16_catalog(M, N, K)
        weight_key = "shuffleweights" if is_shuffle else "weights"
        min_tile_m = min((c["tile_m"] for _, _, c in flydsl_catalog), default=16)
        tasks = []
        for solidx, kernel_name, config in flydsl_catalog:
            if config.get("b_preshuffle", False) != is_shuffle:
                continue
            if config["tile_m"] > max(M, min_tile_m):
                continue
            if N < config["tile_n"] or N % config["tile_n"] != 0:
                continue
            if K % config["split_k"] != 0:
                continue
            ks = K // config["split_k"]
            if ks < config["tile_k"] or ks % config["tile_k"] != 0:
                continue
            if config["split_k"] > 1:
                counters = ((M + config["tile_m"] - 1) // config["tile_m"]) * (
                    N // config["tile_n"]
                )
                if counters > 128:
                    continue
            info = (
                info_keys,
                solidx,
                config["split_k"],
                kernel_name,
                "flydsl",
                is_shuffle,
            )
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                    run_flydsl_gemm_bf16,
                    (["inp", weight_key, "bias"], outdtype, config),
                    dict(run_kwargs),
                    get_gemm_ref,
                    (
                        ["inp", "weights", "bias", "x_scale", "w_scale"],
                        indtype,
                        outdtype,
                    ),
                    {},
                    None,
                    rtol,
                    atol,
                )
            )
        logger.info(f"FlyDSL candidate count for M={M}, N={N}, K={K}: {len(tasks)}")
        return tasks

    def _get_skinny_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        if is_shuffle:
            return []
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        _, _, native_is_skinny = get_native_gemm_funcs()
        if not native_is_skinny(M, N, K, indtype):
            return []
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        info = (info_keys, 2, 0, "sol2", "skinny", is_shuffle)
        return [
            (
                info,
                generate_data,
                (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                run_skinny_gemm_a16w16,
                (["inp", "weights", "bias"], outdtype),
                dict(run_kwargs),
                get_gemm_ref,
                (["inp", "weights", "bias", "x_scale", "w_scale"], indtype, outdtype),
                {},
                None,
                rtol,
                rtol,
            )
        ]

    def _get_torch_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        if is_shuffle:
            return []
        if indtype not in [dtypes.fp16, dtypes.bf16, dtypes.fp8]:
            return []
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        info = (info_keys, 0, 0, "native", "torch", is_shuffle)
        return [
            (
                info,
                generate_data,
                (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                run_torch_gemm_a16w16,
                (["inp", "weights", "bias", "x_scale", "w_scale"], outdtype),
                dict(run_kwargs),
                get_gemm_ref,
                (["inp", "weights", "bias", "x_scale", "w_scale"], indtype, outdtype),
                {},
                None,
                rtol,
                rtol,
            )
        ]

    def _get_triton_tasks(
        self, info_keys, has_bias, indtype, outdtype, scaleAB, is_shuffle, run_kwargs
    ):
        if scaleAB or is_shuffle or outdtype == dtypes.fp32 or indtype != dtypes.bf16:
            return []
        M, N, K = info_keys[2], info_keys[3], info_keys[4]
        rtol = 5e-2 if outdtype == dtypes.bf16 else 1e-2
        info = (info_keys, 0, 0, "auto", "triton", is_shuffle)
        return [
            (
                info,
                generate_data,
                (M, N, K, indtype, outdtype, scaleAB, is_shuffle, 0, has_bias),
                run_triton_gemm_bf16,
                (["inp", "weights", "bias"], outdtype),
                dict(run_kwargs),
                get_gemm_ref,
                (["inp", "weights", "bias", "x_scale", "w_scale"], indtype, outdtype),
                {},
                None,
                rtol,
                rtol,
            )
        ]

    # -------------------------------------------------------------------
    # hipblaslt (delegates to gradlib)
    # -------------------------------------------------------------------

    def _run_hipblaslt(self, ds, args):
        """Run hipblaslt tuning for a single shape via gradlib's Gemm class.

        Gradlib's Gemm uses the same info format:
            (shape_8tuple, solidx, splitK, kernelName, libtype, bpreshuffle)
        We just prepend (gfx, cu_num) to the shape tuple to match self.keys.
        """
        if HipblasltGemm is None:
            logger.warning(
                f"hipblaslt not available, skip. reason: {HIPBLASLT_TUNE_ERROR}"
            )
            return []
        indtype = eval(ds["dtype"])
        outdtype = eval(ds["outdtype"])
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        gemmobj = HipblasltGemm(
            ds["M"],
            ds["N"],
            ds["K"],
            ds["bias"],
            indtype=indtype,
            outdtype=outdtype,
            scaleAB=ds["scaleAB"],
            is_shuffle=ds["bpreshuffle"],
            mp=args.mp,
            err_ratio=args.errRatio,
            profile_file=args.profile_file,
            num_warmup=10,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        rets = gemmobj.run_solutions()
        gemmobj.cleanup()

        result = []
        for info, us, err_ratio in rets:
            shape_tuple, solidx, splitK, kernelName, libtype, bpreshuffle = info
            keys_10 = (gfx, cu_num) + shape_tuple
            result.append(
                (
                    (keys_10, solidx, splitK, kernelName, libtype, bpreshuffle),
                    us,
                    err_ratio,
                )
            )
        return result

    # -------------------------------------------------------------------
    # Main tune loop
    # -------------------------------------------------------------------

    def tune(self, untunedf, tunedf, args):
        libtype = args.libtype
        with_hipblaslt = getattr(args, "with_hipblaslt", False)
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        run_kwargs = {"num_warmup": 10, "num_iters": 101}

        task = []
        tasks_data = []
        hipblaslt_rets = []

        for i in range(len(untunedf)):
            ds = untunedf.loc[i, :]
            indtype = eval(ds["dtype"])
            outdtype = eval(ds["outdtype"])
            outdtype = outdtype if outdtype is not None else indtype
            M, N, K = ds["M"], ds["N"], ds["K"]
            has_bias = ds["bias"]
            scaleAB = ds["scaleAB"]
            is_shuffle = ds["bpreshuffle"]
            info_keys = (
                gfx,
                cu_num,
                M,
                N,
                K,
                has_bias,
                str(indtype),
                str(outdtype),
                scaleAB,
                is_shuffle,
            )
            common = (
                info_keys,
                has_bias,
                indtype,
                outdtype,
                scaleAB,
                is_shuffle,
                run_kwargs,
            )

            prev_count = len(task)
            if "all" in libtype or "asm" in libtype:
                task.extend(self._get_asm_tasks(*common))
            if "all" in libtype or "flydsl" in libtype:
                task.extend(self._get_flydsl_tasks(*common))
            if "all" in libtype or "skinny" in libtype:
                task.extend(self._get_skinny_tasks(*common))
            if "all" in libtype or "torch" in libtype:
                task.extend(self._get_torch_tasks(*common))
            if "all" in libtype or "triton" in libtype:
                task.extend(self._get_triton_tasks(*common))
            if "all" in libtype or "opus" in libtype:
                opus_tasks = self._get_opus_tasks(*common)
                if opus_tasks and _opus_ensure_kids_compiled is not None:
                    opus_kids = {t[0][1] for t in opus_tasks}
                    if _opus_ensure_kids_compiled(opus_kids):
                        logger.info(
                            f"opus subset-compile: expanded sidecar to cover "
                            f"{len(opus_kids)} candidate kids"
                        )
                task.extend(opus_tasks)

            shape_kernel_nums = len(task) - prev_count
            tasks_data.append((shape_kernel_nums, ()))

            if with_hipblaslt and ("all" in libtype or "hipblaslt" in libtype):
                hipblaslt_rets.extend(self._run_hipblaslt(ds, args))

        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                args.mp,
                False,
                args.shape_grouped,
                args.errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        return ret + hipblaslt_rets

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype, _bpreshuffle = info
            if kernelName == "" or pd.isna(kernelName):
                if libtype == "hipblaslt":
                    try:
                        kernelName = aiter.getHipblasltKernelName(int(kernelId))
                    except Exception:
                        kernelName = "None"
                else:
                    kernelName = "None"
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"libtype={libtype} kernelId={kernelId} {kernelName} "
                    f"splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "solidx": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "err_ratio": [err_ratio],
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
    resultList = [
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]
    tuner = GemmA16W16Tuner(
        "GemmA16W16Tuner",
        key,
        resultList,
        description="Tune a16w16 (bf16) GEMM across multiple backends",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
