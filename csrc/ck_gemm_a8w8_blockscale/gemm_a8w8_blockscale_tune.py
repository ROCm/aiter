# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes, logger
from aiter.jit.core import (
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
    get_asm_dir,
)
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import (
    SCALE_TYPE_E8M0,
    SCALE_TYPE_FP32,
    blockscale_bpreshuffle_backends,
)
from aiter.ops.opus.gemm_op_a8w8 import (
    opus_gemm_a8w8_blockscale_bpreshuffle_tune,
)
from aiter.ops.shuffle import (
    shuffle_scale_blockscale_a,
    shuffle_scale_blockscale_b,
    shuffle_weight,
)
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

sys.path.insert(0, str(Path(__file__).parent.parent))
from ck_gemm_a8w8_blockscale_bpreshuffle.gemm_a8w8_blockscale_bpreshuffle_common import (
    kernels_list as candidate_kernels_bpreshuffle_dict,
)

# cktile
from gemm_a8w8_blockscale_cktile_instance import (
    BLOCK_PER_CU_MAX,
    candidate_kernels_cktile_dict,
)
from gemm_a8w8_blockscale_instance import candidate_kernels_dict
from opus_gemm.opus_gemm_common import gfx942_a8w8_kernels_list

try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        fits_shape as fits_shape_flydsl,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        kernels_list as kernels_list_flydsl,
    )
    from aiter.ops.flydsl.utils import is_flydsl_available
except ImportError as _flydsl_import_err:
    print(
        f"[FlyDSL] mxscale preshuffle catalog unavailable "
        f"({_flydsl_import_err}); --libtype flydsl disabled"
    )
    kernels_list_flydsl = {}
    fits_shape_flydsl = None

    def is_flydsl_available():
        return False


block_shape = (128, 128)

FP8_E4M3_MAX = 448.0


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


"""
a8w8_blockscale_gemm tuning for ck, ck_tile and asm
"""


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    """
    Run the reference GEMM operation using PyTorch.
    """

    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))

    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_gemm_a8w8_blockscale_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck_tile type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def run_gemm_a8w8_blockscale(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def run_gemm_a8w8_blockscale_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    zero_bias_buf,
    kernel_name,
    splitK=1,
    preshuffleB=True,
):
    """
    Run gemm a8w8 blockscale tuned kernel for asm type.
    """

    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
        x,
        weight,
        out,
        x_scale,
        w_scale,
        None,
        splitK,
        kernel_name,
        preshuffleB,
        zero_bias_buf,
    )


def run_gemm_a8w8_blockscale_opus(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    kernel_id,
):
    """
    Run gfx942 Opus a8w8 blockscale bpreshuffle tuned kernel.
    """
    return opus_gemm_a8w8_blockscale_bpreshuffle_tune(
        x, weight, x_scale, w_scale, out, kernelId=kernel_id
    )


def run_gemm_a8w8_blockscale_flydsl(
    x,
    weight_shuffle,
    x_scale_shuf,
    w_scale_shuf,
    out,
    kernel_name,
):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        run_gemm_a8w8_mxscale_preshuffle_gfx950,
    )

    return run_gemm_a8w8_mxscale_preshuffle_gfx950(
        x, weight_shuffle, x_scale_shuf, w_scale_shuf, out, kernel_name
    )


def generate_data_e8m0(m, n, k, seed, device="cuda"):
    """Data for the FlyDSL MX backend, kept separate from generate_data()."""
    torch.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)

    def quant(s):
        return fp4_utils.f32_to_mx_e8m0_scale(
            s * FP8_E4M3_MAX, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
        )

    x_scale = quant(torch.rand([m, scale_k], dtype=dtypes.fp32, device=device))
    w_scale = quant(torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device))
    xs = fp4_utils.e8m0_to_f32(x_scale).repeat_interleave(block_shape_k, dim=1)[:, :k]
    ws = (
        fp4_utils.e8m0_to_f32(w_scale)
        .repeat_interleave(block_shape_n, dim=0)
        .repeat_interleave(block_shape_k, dim=1)[:n, :k]
    )
    return {
        "x": x,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        # Scales are shuffled caller-side (like shuffle_weight), so the shuffle is
        # prep, not part of the timed GEMM -- same split the model uses.
        "x_scale_shuf": shuffle_scale_blockscale_a(x_scale, k),
        "w_scale_shuf": shuffle_scale_blockscale_b(w_scale, n, k),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": x.to(dtypes.fp32) * xs,
        "w_deq": weight.to(dtypes.fp32) * ws,
    }


def run_torch_e8m0(x_deq, w_deq, dtype=dtypes.bf16):
    """Reference for the MX backend: operands arrive already dequantized."""
    return F.linear(x_deq, w_deq).to(dtype)


def generate_data(m, n, k, seed, device="cuda"):
    """
    Generate random data for testing the gemm a8w8 blockscale kernel.
    """

    torch.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    zero_bias = torch.zeros((1, n), dtype=torch.float32, device=device)
    return {
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
        "weight_shuffle": weight_shuffle,
        "x_scale_t": x_scale_t,
        "zero_bias": zero_bias,
    }


class GemmA8W8BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}",
        "untune_file": "aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",  # for both results
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    }

    def __init__(self, name, keys, resultList, description=""):
        """
        Initialize the Gemm A8W8 BlockScale Tuner.
        """

        super().__init__(name, keys, resultList, description)

    def run(self, args, fast_mode=False):
        # Stash before super().run() -> pre_process() -> get_untuned_gemm_list().
        self._scaletype_arg = getattr(args, "scaletype", None)
        if not getattr(args, "preshuffle", False):
            self.keys = [k for k in self.keys if k != "scaletype"]
            self.columns = [c for c in self.columns if c != "scaletype"]
            self.sort_keys = [k for k in self.sort_keys if k != "scaletype"]
        if getattr(args, "preshuffle", False):
            self.ARG_DEFAULTS = {
                **self.ARG_DEFAULTS,
                "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
                "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}",
            }
            if args.tune_file == f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}":
                args.tune_file = f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}"
                logger.warning(
                    "--preshuffle without -o: writing to %s (the -o default "
                    "still points at the non-preshuffle family).",
                    args.tune_file,
                )
        return super().run(args, fast_mode)

    def _fill_scaletype(self, df, value):
        if "scaletype" not in self.keys or "scaletype" in df.columns:
            return df
        pos = self.columns.index("scaletype")
        df.insert(min(pos, len(df.columns)), "scaletype", value)
        return df

    def get_untuned_gemm_list(self, untuned_gemm_file):
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        override = getattr(self, "_scaletype_arg", None)
        if override is not None and "scaletype" in self.keys:
            df["scaletype"] = override
            return df
        return self._fill_scaletype(df, SCALE_TYPE_FP32)

    def get_tuned_gemm_list(self, tuned_gemm_file):
        """Mirror of get_untuned_gemm_list: a tuned CSV predating the column is
        all fp32, and pre_process compares the two frames column-for-column."""
        return self._fill_scaletype(
            super().get_tuned_gemm_list(tuned_gemm_file), SCALE_TYPE_FP32
        )

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_CKGEMM_config.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        """
        Setup specific arguments for the tuner.
        """

        self.parser.add_argument(
            "--libtype",
            type=str,
            default="all",
            choices=["ck", "cktile", "asm", "opus", "flydsl", "all", "both"],
            required=False,
            help="CK gemm a8w8 blockscale type to tune: ck, cktile, asm, opus, flydsl, both or all (covers all supported backends across standard/preshuffleB modes)",
        )

        self.parser.add_argument(
            "--scaletype",
            type=str,
            default=None,
            choices=[SCALE_TYPE_FP32, SCALE_TYPE_E8M0],
            required=False,
            help=(
                "Tune the untuned shapes as this scale type, overriding any "
                "scaletype column they carry -- so an ordinary M,N,K shape list "
                "needs no new column to be tuned for e8m0. Defaults to whatever "
                "the input declares, or fp32. The groups never share a pool: an "
                "e8m0 backend takes e8m0 scales in a shuffled layout the fp32 "
                "backends cannot read, and vice versa."
            ),
        )

        self.parser.add_argument(
            "--preshuffle",
            action="store_true",
            help="Enable B-matrix preshuffle for CK gemm a8w8 blockscale",
        )

        self.parser.add_argument(
            "--blockPerCu",
            nargs="+",
            type=int,
            default=list(range(1, BLOCK_PER_CU_MAX + 1)),
            help="List of BlockPerCu values to tune (CKTile only)",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        """
        Calculate performance metrics based on results.
        """

        _info, time, _err_ratio = results
        if time == self.INVALID_TIME or time == self.INF_TIME:
            return 0, 0
        return super().calculate(results, bpes=(1, 1, 2))

    def getKernelName(self, kernelId, libType="ck", preshuffleB=False):
        """
        Get the kernel name based on the kernel ID for different types.
        """
        if libType == "ck":
            kernel_list = (
                candidate_kernels_bpreshuffle_dict
                if preshuffleB
                else candidate_kernels_dict
            )
        elif libType == "cktile":
            # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
            kernel_list = candidate_kernels_cktile_dict
        else:
            return None

        if kernelId >= len(kernel_list) or kernelId < 0:
            return None
        return kernel_list[kernelId].name

    def get_asm_kernels(self, file, preshuffleB):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}

        df = pd.read_csv(file)
        asm_df = (
            df[df["bpreshuffle"] == int(preshuffleB)]
            .reset_index(drop=True)
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            asm_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_gemm_a8w8_blockscale_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        block_per_cu,
        run_kwargs,
    ):
        _gfx, _cu_num, M, N, K, *_ = info_keys
        # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
        kernel_list = {
            k: v
            for k, v in candidate_kernels_cktile_dict.items()
            if v.BlockPerCu in block_per_cu
        }
        gemm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_cktile = []
        for i, kernel in kernel_list.items():
            if not get_gfx().startswith("gfx95") and (
                (kernel.M_Warp * kernel.N_Warp * kernel.K_Warp == 8)
                or (kernel.K_Warp_Tile > 64)  # gfx942 not support
            ):
                continue

            maxsplitK = (
                0
                if preshuffleB
                else (
                    aiter.compute_gemm_SplitK(
                        M,
                        N,
                        K,
                        kernel.M_Tile,
                        kernel.N_Tile,
                        kernel.K_Tile,
                    )
                    if useSplitK
                    else 0
                )
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "cktile", preshuffleB)
                tasks_cktile.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale_cktile,
                        (
                            gemm_keys,
                            i,
                            splitK,
                            preshuffleB,
                        ),
                        dict(run_kwargs),
                        run_torch,
                        (
                            ref_keys,
                            None,
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
        return tasks_cktile

    def get_gemm_a8w8_blockscale_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        _gfx, _cu_num, M, N, K, *_ = info_keys
        kernel_list = (
            candidate_kernels_bpreshuffle_dict
            if preshuffleB
            else candidate_kernels_dict
        )
        kernels_num = len(kernel_list)
        gemm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernel_list[i]
            maxsplitK = (
                0
                if preshuffleB
                else (
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
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "ck", preshuffleB)
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale,
                        (
                            gemm_keys,
                            i,
                            splitK,
                            preshuffleB,
                        ),
                        dict(run_kwargs),
                        run_torch,
                        (
                            ref_keys,
                            None,
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

    def get_gemm_a8w8_blockscale_opus_tune_task(
        self,
        info_keys,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        gfx, _, M, N, K, *_ = info_keys
        if not preshuffleB or gfx != "gfx942":
            return []

        gemm_keys = ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        ref_args = (ref_keys, None, dtypes.bf16)
        tasks_opus = []
        for kernel_id, kernel in gfx942_a8w8_kernels_list.items():
            if N % kernel.B_N != 0 or K % kernel.B_K != 0:
                continue
            if not kernel.has_oob and M % kernel.B_M != 0:
                continue
            info = (info_keys, kernel_id, 0, kernel.name, "opus", preshuffleB)
            gemm_args = (gemm_keys, kernel_id)
            tasks_opus.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed),
                    run_gemm_a8w8_blockscale_opus,
                    gemm_args,
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
        return tasks_opus

    def get_gemm_a8w8_blockscale_flydsl_tune_task(
        self,
        info_keys,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        gfx, _, M, N, K, *_ = info_keys
        if not preshuffleB or gfx != "gfx950":
            return []
        if not kernels_list_flydsl or fits_shape_flydsl is None:
            return []
        if not is_flydsl_available():
            return []

        gemm_keys = ["x", "weight_shuffle", "x_scale_shuf", "w_scale_shuf", "out"]
        ref_args = (["x_deq", "w_deq"], dtypes.bf16)
        tasks_flydsl = []
        for kernel_id, ki in kernels_list_flydsl.items():
            if (ki.a_dtype, ki.b_dtype) != ("fp8", "fp8"):
                continue
            if not fits_shape_flydsl(ki, M, N, K):
                continue
            # kernelName carries the full launch config (incl. split-K), so
            # dispatch never needs kernelId -- it is recorded for reference only.
            info = (info_keys, kernel_id, ki.split_k, ki.name, "flydsl", preshuffleB)
            tasks_flydsl.append(
                (
                    info,
                    generate_data_e8m0,
                    (M, N, K, seed),
                    run_gemm_a8w8_blockscale_flydsl,
                    (gemm_keys, ki.name),
                    dict(run_kwargs),
                    run_torch_e8m0,
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
        return tasks_flydsl

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import (
            gemm_a8w8_blockscale,
            gemm_a8w8_blockscale_bpreshuffle,
        )
        from aiter.test_common import checkAllclose, run_perftest

        is_preshuffle = args.preshuffle
        untunedf = self.untunedf
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            shape_str = f"({M}, {N}, {K})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                row_scaletype = (
                    row["scaletype"] if "scaletype" in row.index else SCALE_TYPE_FP32
                )
                is_e8m0_row = is_preshuffle and row_scaletype == SCALE_TYPE_E8M0
                if is_e8m0_row:
                    gd = generate_data_e8m0(M, N, K, 0)
                    out, us = run_perftest(
                        gemm_a8w8_blockscale_bpreshuffle,
                        gd["x"],
                        gd["weight_shuffle"],
                        gd["x_scale_shuf"],
                        gd["w_scale_shuf"],
                        **run_kwargs,
                    )
                    ref = run_torch_e8m0(gd["x_deq"], gd["w_deq"])
                else:
                    gd = generate_data(M, N, K, 0)
                    x, weight, x_scale, w_scale, out = (
                        gd["x"],
                        gd["weight"],
                        gd["x_scale"],
                        gd["w_scale"],
                        gd["out"],
                    )
                    weight_shuffle, x_scale_t = gd["weight_shuffle"], gd["x_scale_t"]
                    if is_preshuffle:
                        out, us = run_perftest(
                            gemm_a8w8_blockscale_bpreshuffle,
                            x,
                            weight_shuffle,
                            x_scale_t,
                            w_scale,
                            **run_kwargs,
                        )
                    else:
                        out, us = run_perftest(
                            gemm_a8w8_blockscale,
                            x,
                            weight,
                            x_scale,
                            w_scale,
                            **run_kwargs,
                        )
                    ref = run_torch(x, weight, x_scale, w_scale)
                err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
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

    def get_gemm_a8w8_blockscale_asm_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        _gfx, _cu_num, M, N, K, *_ = info_keys
        asm_kernel_list_csv = (
            f"{get_asm_dir()}/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv"
        )
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv, preshuffleB)
        if not asm_kernels:
            return []

        gemm_asm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out", "zero_bias"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out", "zero_bias"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_asm = []
        asm_kernel_id = 0
        for key, kernel_names in asm_kernels.items():
            _tile_m, tile_n, splitk_supported = key
            # Respect ASM kernel tile constraints from the config CSV.
            if N % tile_n != 0:
                continue
            splitK_list = (
                get_valid_asm_splitK_list(K, 8)
                if useSplitK and int(splitk_supported) == 1
                else [1]
            )
            for kernel_name in kernel_names:
                for splitK in splitK_list:
                    info = (
                        info_keys,
                        asm_kernel_id,
                        splitK,
                        kernel_name,
                        "asm",
                        preshuffleB,
                    )
                    tasks_asm.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a8w8_blockscale_asm,
                            (
                                gemm_asm_keys,
                                kernel_name,
                                splitK,
                                preshuffleB,
                            ),
                            dict(run_kwargs),
                            run_torch,
                            (
                                ref_keys,
                                None,
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
                    asm_kernel_id += 1
        return tasks_asm

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        isPreshuffleB = args.preshuffle
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        block_per_cu = args.blockPerCu
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 0
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            row_scaletype = (
                str(untunedf.loc[i, "scaletype"])
                if "scaletype" in untunedf.columns
                else SCALE_TYPE_FP32
            )
            prev_task_count = len(task)
            info_keys = (gfx, cu_num, M, N, K, row_scaletype)
            lib = args.libtype
            row_backends = blockscale_bpreshuffle_backends(row_scaletype)

            def _in_group(libtype, _b=row_backends):
                """Backends only compete inside their own scale-type group."""
                return libtype in _b

            if lib in ("ck", "both", "all") and _in_group("ck"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("cktile", "both", "all") and _in_group("cktile"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_cktile_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        block_per_cu,
                        run_kwargs,
                    )
                )
            if lib in ("asm", "all") and _in_group("asm"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_asm_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("opus", "all") and _in_group("opus"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_opus_tune_task(
                        info_keys,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("flydsl", "all") and _in_group("flydsl"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_flydsl_tune_task(
                        info_keys,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
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
        """
        post-process the tuning results into a DataFrame.
        """

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype, preshuffleB = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
                else (
                    self.getKernelName(kernelId, libtype, preshuffleB)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} "
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


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K", "scaletype"]
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
    tuner = GemmA8W8BlockScaleTuner(
        "GemmA8W8BlockScaleTuner",
        key,
        resultList,
        description="Tune a8w8 blockscale GEMM (CK, CKTile, ASM backends)",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
