# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
import pandas as pd
import os
import sys
from aiter import QuantType
from aiter.jit.core import (
    get_asm_dir,
    AITER_CSRC_DIR,
    AITER_CONFIG_FMOE,
    AITER_ROOT_DIR,
)
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    asm_stage1,
    torch_moe_stage1,
    torch_moe_stage2,
    torch_moe,
    cktile_moe_stage1,
    cktile_moe_stage2,
)
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, dtype2str_dict
from aiter.ops.shuffle import (
    shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
    pack_int8_to_packed_int4,
    shuffle_scale_for_int4,
)
from aiter.utility.mp_tuner import mp_tuner
from aiter.int4_utils import (
    rearrange_4bit_elements,
    convert_int8_to_uint32_int4,
)
from aiter import dtypes
from aiter import ActivationType as ActivationType
from aiter.jit.utils.chip_info import get_gfx
import torch.nn.functional as F
from einops import rearrange
from aiter.utility.base_tuner import TunerCommon
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import moe_mxfp4_sort


from aiter.ops.flydsl.utils import is_flydsl_available

if is_flydsl_available():
    from aiter.ops.flydsl.moe_kernels import (
        get_flydsl_stage1_kernels,
        get_flydsl_stage2_kernels,
        get_flydsl_stage1_kernels_int4_bf16,
        get_flydsl_stage2_kernels_int4_bf16,
        flydsl_moe_stage1,
        flydsl_moe_stage2,
    )

sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


FLYDSL_FALLBACK_TAG = "flydsl_fallback"
TUNE_MOE_EXPERT_BALANCE = (
    os.environ.get("TUNE_MOE_EXPERT_BALANCE", "False").lower() == "true"
)

COS_DIFF_THRESHOLD = 1e-1


def torch_dynamic_mxfp8_quant(x: torch.Tensor):
    """MXFP8 quantization (e4m3fn + e8m0 block scale, block=32).

    Same numerics as ``aiter/bench_stage2_a8w4.py`` for a8w4 activations.
    """
    BLOCK = 32
    orig_shape = x.shape
    x_f32 = x.reshape(-1, x.shape[-1] // BLOCK, BLOCK).float()

    amax, _ = torch.max(torch.abs(x_f32), dim=-1)
    amax_i32 = amax.view(torch.int32)
    amax_rounded = (amax_i32 + 0x200000) & 0xFF800000
    exp_field = (amax_rounded >> 23) & 0xFF

    e8m0_biased = torch.clamp(exp_field - 8, min=0)
    quant_exp = 254 - e8m0_biased
    quant_scale = (quant_exp << 23).view(torch.float32)

    scaled = x_f32 * quant_scale.unsqueeze(-1)
    fp8_vals = scaled.to(torch.float8_e4m3fn)
    fp8_bytes = fp8_vals.view(torch.uint8)

    e8m0_bytes = e8m0_biased.to(torch.uint8).view(dtypes.fp8_e8m0)
    return fp8_bytes.view(*orig_shape), e8m0_bytes.view(
        *orig_shape[:-1], orig_shape[-1] // BLOCK
    )


def cosine_diff_compare(ref, res, msg="", printLog=True):
    from aiter import logger

    x = ref.double().flatten()
    y = res.double().flatten()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    if printLog:
        if cos_diff < COS_DIFF_THRESHOLD:
            logger.info(f"{msg}[cosine_diff={cos_diff:.6f} \033[32mpassed~\033[0m]")
        else:
            logger.info(f"{msg}[cosine_diff={cos_diff:.6f} \033[31mfailed!\033[0m]")
    return cos_diff if cos_diff >= COS_DIFF_THRESHOLD else 0.0


class FmoeTuner(TunerCommon):
    ARG_DEFAULTS = {
        **TunerCommon.ARG_DEFAULTS,
        "verbose": False,
        "tune_file": f"{AITER_CONFIG_FMOE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/untuned_fmoe.csv",
        "errRatio": 0.5,
        "batch": 100,
        "profile_file": "",  # for all results
        "config_env_name": "AITER_CONFIG_FMOE",
    }

    def _clear_op_caches(self):
        try:
            import aiter.fused_moe as fmoe_module

            if hasattr(fmoe_module, "cfg_2stages"):
                fmoe_module.cfg_2stages = None
            if hasattr(fmoe_module, "get_2stage_cfgs"):
                fmoe_module.get_2stage_cfgs.cache_clear()
        except ImportError:
            pass

    def _setup_specific_arguments(self):

        self.parser.add_argument(
            "--last",
            action="store_true",
            required=False,
            help="Only last kernel is tuned, if not, only kernels that are not in the tuned_fmoe.csv are tuned",
        )

    @staticmethod
    def weight_quant(
        weight,
        qType,
        quant_dtype,
    ):
        E, dim1, dim2 = weight.shape
        if qType == aiter.QuantType.per_Tensor and quant_dtype != torch.int4:
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight.view(E, -1), quant_dtype=quant_dtype
            )
        elif qType == QuantType.per_1x128:
            weight_qt = (
                weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, -1, 128 * 128)
            )
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_qt, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(E, -1)
            weight_qt = (
                weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .view(E, dim1, dim2)
            )
        elif (
            qType == aiter.QuantType.per_Tensor and quant_dtype == torch.int4
        ):  # int4 w quant
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight.view(E, -1), quant_dtype=dtypes.i8, dtypeMax=7
            )
        elif (
            qType == aiter.QuantType.per_Token and quant_dtype == torch.int4
        ):  # int4 w quant
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight, quant_dtype=dtypes.i8, dtypeMax=7
            )
        elif qType == QuantType.per_1x32 and quant_dtype == dtypes.i4x2:
            # a16wi4: groupwise quantize to int4 [-7, 7] with group_size=32
            group_size = 32
            w_groups = weight.view(E, dim1, dim2 // group_size, group_size)
            w_group_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
            w_scale_raw = (w_group_max / 7.0).squeeze(-1)  # [E, dim1, G]
            weight_qt = (
                (w_groups / w_scale_raw.unsqueeze(-1))
                .round()
                .clamp(-7, 7)
                .to(dtypes.i8)
            )
            weight_qt = weight_qt.view(E, dim1, dim2)
            # Scale: [E, K//32, N] = permuted from [E, N, G]
            weight_scale = w_scale_raw.permute(0, 2, 1).to(dtypes.bf16)  # [E, G, N]
            return weight_qt, weight_scale
        else:
            torch_quant = aiter.get_torch_quant(qType)
            weight_qt, weight_scale = torch_quant(weight, quant_dtype=quant_dtype)
        return weight_qt, weight_scale

    def get_kernels_dict(self, file, key="tile_m"):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}
        df = pd.read_csv(file)
        kernel_dict = df.groupby(key)["knl_name"].apply(list).to_dict()
        return kernel_dict

    @staticmethod
    def ck_moe_stage1_fwd_out(
        a1_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w1_scale,
        a1_scale,
        dtype,
        topk,
        kernelName,
        blockM,
        q_type,
        act_type,
        splitk=0,
    ):
        inter_dim = w1_qt_shffle_ck.shape[1] // 2
        token_num = a1_qt.shape[0]
        is_splitk = q_type == QuantType.per_1x128 and splitk > 1
        if is_splitk:
            sorted_size = min(token_num * topk * blockM, sorted_ids.shape[0])
            tmp_out = torch.empty(
                (sorted_size, w1_qt_shffle_ck.shape[1]),
                dtype=dtypes.fp32,
                device=a1_qt.device,
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim),
                dtype=dtype,
                device=a1_qt.device,
            )
            tmp_out = out
        try:
            ck_moe_stage1_fwd(
                a1_qt,
                w1_qt_shffle_ck,
                w2_qt_shffle_ck,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                tmp_out,
                topk,
                kernelName,
                w1_scale,
                a1_scale,
                blockM,
                sorted_weights,
                q_type,
                act_type,
                splitk if is_splitk else 0,
                dst_type=dtype if is_splitk else None,
            )
        except Exception:
            raise
        if is_splitk:
            out = torch.empty(
                (token_num, topk, inter_dim),
                dtype=dtype,
                device=a1_qt.device,
            )
            valid_out = tmp_out[: token_num * topk, :]
            if act_type == ActivationType.Silu or (
                isinstance(act_type, str) and "silu" in act_type.lower()
            ):
                aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
            else:
                aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
        if q_type == QuantType.per_1x128:
            quant_func = aiter.get_hip_quant(q_type)
            a2, a2_scale = quant_func(
                out,
                quant_dtype=a1_qt.dtype,
            )
            out = a2
        return out

    @staticmethod
    def ck_moe_stage2_fwd_out(
        a2_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w2_scale,
        a2_scale,
        dtype,
        topk,
        kernelName,
        blockM,
        q_type,
        act_type,
    ):
        model_dim = w2_qt_shffle_ck.shape[1]
        token_num = a2_qt.shape[0]

        out = torch.zeros(
            (token_num, model_dim),
            dtype=dtype,
            device=a2_qt.device,
        )
        return ck_moe_stage2_fwd(
            a2_qt,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            kernelName,
            w2_scale,
            a2_scale,
            blockM,
            sorted_weights,
            q_type,
            act_type,
        )

    @staticmethod
    def cktile_moe_stage1_out(
        a1_fp8,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w1_scale_aiter,
        bias,
        dtype,
        topk,
        blockM,
        act_type,
    ):
        M_sorted = sorted_ids.shape[0]
        model_dim = a1_fp8.shape[1]
        a1_scale = torch.ones(
            (M_sorted, model_dim // 32), dtype=dtypes.fp8_e8m0, device=a1_fp8.device
        )
        return cktile_moe_stage1(
            a1_fp8,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            None,
            topk,
            blockM,
            a1_scale=a1_scale,
            w1_scale=w1_scale_aiter.view(dtypes.fp8_e8m0),
            sorted_weights=sorted_weights,
            bias1=bias,
            activation=act_type,
            split_k=1,
            dtype=dtype,
        )

    @staticmethod
    def cktile_moe_stage2_out(
        a2_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w2_scale_aiter,
        a2_scale_sort,
        bias,
        dtype,
        topk,
        blockM,
        act_type,
    ):
        token_num = a2_qt.shape[0]
        model_dim = w2_qt_shffle_ck.shape[1]
        out = torch.zeros(
            (token_num, model_dim),
            dtype=dtype,
            device=a2_qt.device,
        )
        return cktile_moe_stage2(
            a2_qt,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            w2_scale=w2_scale_aiter.view(dtypes.fp8_e8m0),
            a2_scale=a2_scale_sort,
            block_m=blockM,
            activation=act_type,
            sorted_weights=sorted_weights,
            bias2=bias,
        )

    @staticmethod
    def run_flydsl_stage1_out(
        a1_qt,
        w1_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w1_scale_aiter,
        a1_scale,
        bias,
        dtype,
        topk,
        kparams,
        blockM,
        q_dtype_a,
        q_type,
        act_type,
    ):
        act = "swiglu" if act_type == ActivationType.Swiglu else "silu"
        a_scale_one = kparams.get("a_scale_one", False)
        _out_dtype = kparams["out_dtype"]
        token_num = a1_qt.shape[0]
        inter_dim = w1_qt_shffle_ck.shape[1] // 2
        result = flydsl_moe_stage1(
            a=a1_qt.to(dtypes.fp8) if q_dtype_a == dtypes.fp8 else a1_qt,
            w1=w1_qt_shffle_ck,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            topk=topk,
            tile_m=kparams["tile_m"],
            tile_n=kparams["tile_n"],
            tile_k=kparams["tile_k"],
            a_dtype=kparams["a_dtype"],
            b_dtype=kparams["b_dtype"],
            out_dtype=_out_dtype,
            act=act,
            w1_scale=w1_scale_aiter,
            a1_scale=a1_scale,
            sorted_weights=sorted_weights,
            use_async_copy=True,
            k_batch=kparams.get("k_batch", 1),
            waves_per_eu=kparams.get("waves_per_eu", 3),
            b_nt=kparams.get("b_nt", 2),
            gate_mode=kparams.get("gate_mode", "separated"),
            a_scale_one=a_scale_one,
            xcd_swizzle=kparams.get("xcd_swizzle", 0),
            bias=bias,
        )
        if isinstance(result, tuple):
            out_raw = result[0]
            if _out_dtype == "fp4":
                total_fp4_bytes = token_num * topk * (inter_dim // 2)
                fp4_flat = out_raw.view(-1).view(torch.uint8)[:total_fp4_bytes]
                return fp4_flat.view(dtypes.fp4x2).reshape(token_num, topk, -1)
            else:
                # fuse_fp8: out_raw is fp8 tensor, shape (token_num, topk, inter_dim)
                return out_raw.reshape(token_num, topk, -1)
        return result

    @staticmethod
    def run_flydsl_stage2_out(
        a2_qt,
        w2_shuffled_flydsl,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w2_scale_shuffled_flydsl,
        a2_scale,
        moe_buf,
        bias,
        dtype,
        topk,
        kparams,
        blockM,
        q_type,
        act_type,
    ):
        if kparams.get("mode", "atomic") == "atomic":
            moe_buf.zero_()

        sort_block_m = kparams.get("sort_block_m", 0)
        persist = kparams.get("persist", None)
        return flydsl_moe_stage2(
            inter_states=a2_qt,
            w2=w2_shuffled_flydsl,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=moe_buf,
            topk=topk,
            tile_m=kparams["tile_m"],
            tile_n=kparams["tile_n"],
            tile_k=kparams["tile_k"],
            a_dtype=kparams["a_dtype"],
            b_dtype=kparams["b_dtype"],
            out_dtype=kparams["out_dtype"],
            mode=kparams.get("mode", "atomic"),
            w2_scale=w2_scale_shuffled_flydsl,
            a2_scale=a2_scale,
            sorted_weights=sorted_weights,
            sort_block_m=sort_block_m,
            persist=persist,
            b_nt=kparams.get("b_nt", 0),
            xcd_swizzle=kparams.get("xcd_swizzle", 0),
            bias=bias,
        )

    @staticmethod
    def run_asm_stage1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        out,
        a1_scale,
        w1_scale,
        topk,
        block_m,
        kernelName,
        ksplit,
        activation,
        quant_type,
        doweight_stage1,
    ):
        if not doweight_stage1:
            sorted_weights = None
        asm_stage1(
            input,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            block_m,
            kernelName,
            ksplit,
            activation,
            quant_type,
            a1_scale,
            w1_scale,
            sorted_weights,
        )
        return out

    # do weight at stage1
    @staticmethod
    def run_1stage_fmoe_g1u1_tkw1(
        hidden_states,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a1_scale,
        w1_scale,
        w2_scale,
        fc2_smooth_scale=None,
        quant_type=QuantType.No,
        isG1U1=False,
        activation=ActivationType.Silu,
        kernel_name="",
        topk=2,
        dtype=dtypes.bf16,
    ):
        moe_buf = torch.zeros(
            (a1.shape[0], a1.shape[1]),
            dtype=dtype,
            device="cuda",
        )
        aiter.fmoe_g1u1_tkw1(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernel_name,
            fc2_smooth_scale=fc2_smooth_scale,
            activation=activation,
        )
        return moe_buf

    @staticmethod
    def run_1stage_fmoe_fp8_blockscale_g1u1(
        hidden_states,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a1_scale,
        w1_scale,
        w2_scale,
        fc2_smooth_scale=None,
        quant_type=QuantType.No,
        isG1U1=False,
        activation=ActivationType.Silu,
        kernel_name="",
        topk=2,
        dtype=dtypes.bf16,
    ):
        moe_buf = torch.zeros(
            (a1.shape[0], a1.shape[1]),
            dtype=dtype,
            device="cuda",
        )
        aiter.fmoe_fp8_blockscale_g1u1(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernel_name,
            fc2_smooth_scale=fc2_smooth_scale,
            activation=activation,
        )
        return moe_buf

    @staticmethod
    def run_1stage_fmoe_g1u1(
        hidden_states,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        a1_scale,
        w1_scale,
        w2_scale,
        fc2_smooth_scale=None,
        quant_type=QuantType.No,
        isG1U1=False,
        activation=ActivationType.Silu,
        kernel_name="",
        topk=2,
        dtype=dtypes.bf16,
    ):
        moe_buf = torch.zeros(
            (a1.shape[0], a1.shape[1]),
            dtype=dtype,
            device="cuda",
        )
        aiter.fmoe_g1u1(
            moe_buf,
            a1,
            w1,
            w2,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            a1_scale,
            w1_scale,
            w2_scale,
            kernel_name,
            fc2_smooth_scale=fc2_smooth_scale,
            activation=activation,
        )
        return moe_buf

    @staticmethod
    def get_1stage_fmoe_func(
        quant_type, q_dtype_a, activation, isG1U1, doweight_stage1
    ):
        fmoe_func = None
        if (
            quant_type == QuantType.No
            and activation == ActivationType.Silu
            and not isG1U1
            or quant_type == QuantType.per_1x32
        ):
            print("not support No Quant Silu G1U0 1 stage or per_1x32 quant tuning!")
        else:
            if quant_type == QuantType.per_1x128:
                fmoe_func = FmoeTuner.run_1stage_fmoe_fp8_blockscale_g1u1
            elif (q_dtype_a == dtypes.fp8) and doweight_stage1:
                fmoe_func = FmoeTuner.run_1stage_fmoe_g1u1_tkw1
            elif isG1U1:
                fmoe_func = FmoeTuner.run_1stage_fmoe_g1u1

        return fmoe_func

    @staticmethod
    def generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device="cuda",
    ):
        torch.manual_seed(0)
        input = torch.randn((token, model_dim), dtype=dtype) / 10
        if use_g1u1:
            w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
        else:
            w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype) / 10
        w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
        w1_qt, w1_scale = FmoeTuner.weight_quant(w1, q_type, quant_dtype=q_dtype_w)
        w2_qt, w2_scale = FmoeTuner.weight_quant(w2, q_type, quant_dtype=q_dtype_w)
        if q_dtype_w is not dtypes.fp4x2:
            w1_qt = w1_qt.view(w1.shape)
            w2_qt = w2_qt.view(w2.shape)
        else:
            w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
            w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
        if TUNE_MOE_EXPERT_BALANCE:
            score = torch.zeros((token, expert), dtype=dtype)
            start_col = 0
            end_col = topk
            for token_id in range(token):
                score[token_id, start_col:end_col] = 1.0
                start_col = end_col % expert
                end_col = start_col + topk
        else:
            score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)
        if q_type == QuantType.per_1x128:
            a1_qt, a1_scale = aiter.pertoken_quant(
                input.view(token, -1, 128), quant_dtype=q_dtype_a
            )
            a1_qt = a1_qt.view(token, model_dim)
            a1_scale = a1_scale.squeeze(-1)
        elif (
            q_type == aiter.QuantType.per_1x32
            and (q_dtype_a in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
            and q_dtype_w == dtypes.fp4x2
        ):  # a16w4 or a8w4
            a1_qt = input.to(dtype)
            a1_scale = None
        elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:  # a16wi4
            a1_qt = input.to(dtypes.bf16)
            a1_scale = None
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)
        del w1, w2, score
        if q_dtype_w is not dtypes.fp4x2:
            w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
            w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))
        else:
            w1_qt_shffle = w1_qt
            w2_qt_shffle = w2_qt

        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
        )
        needed = sorted_expert_ids.shape[0] * blockM
        if sorted_ids.shape[0] < needed:
            pad = torch.full(
                (needed - sorted_ids.shape[0],),
                token,
                dtype=sorted_ids.dtype,
                device=sorted_ids.device,
            )
            sorted_ids = torch.cat([sorted_ids, pad])
            sorted_weights = torch.cat(
                [
                    sorted_weights,
                    torch.zeros(
                        pad.shape[0],
                        dtype=sorted_weights.dtype,
                        device=sorted_weights.device,
                    ),
                ]
            )
        return (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        )

    @staticmethod
    def generate_asm_stage1(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
        blockM,
        device="cuda",
    ):
        (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        ) = FmoeTuner.generate_data(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            blockM,
            device,
        )
        if q_type == QuantType.per_1x128:
            ratio = a1_scale.element_size() // a1_qt.element_size()
            out1 = torch.zeros(
                (token + (token * ratio + 127) // 128, topk, inter_dim),
                dtype=a1_qt.dtype,
            )
        else:
            out1 = torch.empty(
                (token, topk, inter_dim),
                dtype=dtype,
            )
        a1_scale_t = a1_scale
        if q_type == QuantType.per_1x128:
            a1_scale_t = a1_scale.t().contiguous()
        return (
            a1_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            out1,
            a1_scale_t,
            w1_scale,
            topk_weights,
            topk_ids,
            w1_qt,
            w2_qt,
            a1_scale,
        )

    @staticmethod
    def generate_data_2stages(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
        blockM,
        stage=1,
        device="cuda",
    ):
        (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        ) = FmoeTuner.generate_data(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            blockM,
            device,
        )
        # Pre-bind so branches that skip shuffle_scale_* still reach `is None` below.
        w1_scale_aiter = None
        w2_scale_aiter = None
        if q_dtype_w == torch.int4 and q_type != QuantType.per_1x32:
            w1_qt_shffle_ck = rearrange_4bit_elements(
                convert_int8_to_uint32_int4(
                    shuffle_weight(w1_qt, (16, 16), use_int4=True)
                )
            )
            w2_qt_shffle_ck = rearrange_4bit_elements(
                convert_int8_to_uint32_int4(
                    shuffle_weight(w2_qt, (16, 16), use_int4=True)
                )
            )
        elif q_dtype_w == dtypes.fp4x2 and q_dtype_a == dtypes.fp4x2:
            w1_qt_shffle_ck = shuffle_weight(w1_qt, (16, 16))
            w2_qt_shffle_ck = shuffle_weight(w2_qt, (16, 16))
        elif q_dtype_w == dtypes.fp4x2 and q_dtype_a == dtypes.fp8:
            # a8w4 per_1x32 stage1 just support tune a1_cast now.
            w1_qt_shffle_ck = shuffle_weight_a16w4(w1_qt, 16, True)
            w1_scale_aiter = shuffle_scale_a16w4(w1_scale, expert, True)
            w2_qt_shffle_ck = shuffle_weight_a16w4(w2_qt, 16, False)
            w2_scale_aiter = shuffle_scale_a16w4(w2_scale, expert, False)
        else:
            w1_qt_shffle_ck = w1_qt_shffle
            w2_qt_shffle_ck = w2_qt_shffle
        if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
            # a16wi4 int4 FlyDSL path: pack int8 -> int4, shuffle scales
            E1 = w1_qt.shape[0]
            N1, K1 = w1_qt.shape[1], w1_qt.shape[2]
            w1_qt_shffle_flydsl = pack_int8_to_packed_int4(
                shuffle_weight(w1_qt, (16, 16))
            ).view(E1, N1, K1 // 2)
            E2 = w2_qt.shape[0]
            N2, K2 = w2_qt.shape[1], w2_qt.shape[2]
            w2_qt_shffle_flydsl = pack_int8_to_packed_int4(
                shuffle_weight(w2_qt, (16, 16))
            ).view(E2, N2, K2 // 2)
            w1_scale_flydsl = (
                shuffle_scale_for_int4(w1_scale, group_size=32).view(-1).contiguous()
            )
            w2_scale_flydsl = (
                shuffle_scale_for_int4(w2_scale, group_size=32).view(-1).contiguous()
            )
            w1_qt_shffle_ck = w1_qt_shffle
            w2_qt_shffle_ck = w2_qt_shffle
            w1_scale_aiter = w1_scale
            w2_scale_aiter = w2_scale
        else:
            if w1_scale_aiter is None:
                w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
                w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

            w1_qt_shffle_flydsl = w1_qt_shffle_ck
            w2_qt_shffle_flydsl = w2_qt_shffle_ck
            w1_scale_flydsl = w1_scale_aiter
            w2_scale_flydsl = w2_scale_aiter

        if stage == 1:
            if not doweight_stage1:
                sorted_weights = None
            if (
                q_type == QuantType.per_1x32
                and q_dtype_w != dtypes.i4x2
                and q_dtype_a == dtypes.fp4x2
            ):
                a1_scale_fp4_sort = moe_mxfp4_sort(
                    a1_scale,  # a1_scale[: token * topk, :].view(token, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=max(32, blockM),
                )
            else:
                a1_scale_fp4_sort = a1_scale

            # For the _fp8 FlyDSL variant (a_scale_one=True): cast bf16 input to fp8.
            a1_qt_fp8_cast = input.to(dtypes.fp8)

            return (
                a1_qt,  # 0
                w1_qt_shffle_ck,  # 1
                w2_qt_shffle_ck,  # 2
                a1_scale,  # 3
                w1_scale,  # 4
                sorted_ids,  # 5
                sorted_expert_ids,  # 6
                sorted_weights,  # 7
                num_valid_ids,  # 8
                moe_buf,  # 9
                w1_qt,  # 10
                w2_qt,  # 11
                topk_weights,  # 12
                topk_ids,  # 13
                a1_scale_fp4_sort,  # 14
                w1_scale_aiter,  # 15
                w1_qt_shffle_flydsl,  # 16
                w2_qt_shffle_flydsl,  # 17
                w1_scale_flydsl,  # 18
                w2_scale_flydsl,  # 19
                a1_qt_fp8_cast,  # 20 — fp8-cast input for _fp8 FlyDSL variant
                None,  # 21 — None placeholder (a1_scale=None for a8w4 torch ref)
                (
                    torch.clamp(
                        torch.randn(
                            (expert, inter_dim * 2), dtype=dtype, device=device
                        ),
                        -1.0,
                        1.0,
                    ).to(torch.float32)
                    if (
                        act_type == ActivationType.Swiglu
                        and q_type == QuantType.per_1x32
                        and q_dtype_a == dtypes.fp8
                        and dtype in [dtypes.bf16, dtypes.fp16]
                    )
                    else None
                ),  # 22 — bias for stage1 (a8w4 only)
            )
        elif stage == 2:
            # a8w4: a1_scale is dummy non-None → torch_moe_stage1's per_1x32
            # branch would call mxfp4_to_f32(bf16), pass None to take a16w4 path.
            ref_a1_scale = (
                None
                if (q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp8)
                else a1_scale
            )
            ref1 = FmoeTuner.run_torch_moe_stage1(
                a1_qt,
                w1_qt,
                w2_qt,
                topk_weights,
                topk_ids,
                a1_scale=ref_a1_scale,
                w1_scale=w1_scale,
                dtype=dtype,
                activation=act_type,
                quant_type=q_type,
                doweight_stage1=doweight_stage1,
                topk=topk,
            )
            # ref1 is always bf16
            ref1_bf16 = ref1

            if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
                # a16wi4: bf16 passthrough, no inter-stage quant
                a2_qt = ref1
                a2_scale = None
                a2_scale_mxfp4_sort = None
            elif q_type == QuantType.per_1x128:
                ref1, ref_scale = aiter.pertoken_quant(
                    ref1.view(ref1.shape[0], -1, 128), quant_dtype=q_dtype_a
                )
                ref1 = ref1.view(ref1.shape[0], topk, -1)
                ref_scale = ref_scale.view(token, -1)
                a2_qt = ref1
                a2_scale = ref_scale
                a2_scale_mxfp4_sort = a2_scale
            elif q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp4x2:
                torch_quant = aiter.get_torch_quant(q_type)
                a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
                a2_scale_mxfp4_sort = moe_mxfp4_sort(
                    a2_scale[: token * topk, :].view(token, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=blockM,
                )
            elif q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp8:
                # FlyDSL stage2 receives fp8 input
                a2_qt = ref1.to(dtypes.fp8)
                M = sorted_ids.shape[0]
                N = a2_qt.shape[-1]
                a2_scale = torch.ones(
                    [token * topk, N // 32], dtype=dtypes.fp8_e8m0, device=a2_qt.device
                )
                a2_scale_mxfp4_sort = torch.ones(
                    [M, N // 32], dtype=dtypes.fp8_e8m0, device=a2_qt.device
                )
            else:
                torch_quant = aiter.get_torch_quant(q_type)
                a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
                a2_scale_mxfp4_sort = a2_scale
            a2_qt = a2_qt.view(token, topk, -1)
            if doweight_stage1:
                sorted_weights = None

            return (
                a2_qt,  # 0  — fp8 for FlyDSL (a8w4), fp4x2 for a4w4
                w1_qt_shffle_ck,  # 1
                w2_qt_shffle_ck,  # 2
                a2_scale,  # 3
                w2_scale,  # 4
                sorted_ids,  # 5
                sorted_expert_ids,  # 6
                sorted_weights,  # 7
                num_valid_ids,  # 8
                moe_buf,  # 9
                w1_qt,  # 10
                w2_qt,  # 11
                topk_weights,  # 12
                topk_ids,  # 13
                a2_scale_mxfp4_sort,  # 14
                w2_scale_aiter,  # 15
                w1_qt_shffle_flydsl,  # 16
                w2_qt_shffle_flydsl,  # 17
                w1_scale_flydsl,  # 18
                w2_scale_flydsl,  # 19
                ref1_bf16,  # 20 — bf16 stage1 output for torch ref (a8w4)
                None,  # 21 — None placeholder (a2_scale=None for a8w4 torch ref)
                (
                    torch.clamp(
                        torch.randn((expert, model_dim), dtype=dtype, device=device),
                        -1.0,
                        1.0,
                    ).to(torch.float32)
                    if (
                        act_type == ActivationType.Swiglu
                        and q_type == QuantType.per_1x32
                        and q_dtype_a == dtypes.fp8
                        and dtype in [dtypes.bf16, dtypes.fp16]
                    )
                    else None
                ),  # 22 — bias for stage2 (a8w4 only)
            )

    @staticmethod
    def generate_data_1stage(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM=32,
        device="cuda",
    ):
        (
            input,
            a1_qt,
            w1_qt,
            w2_qt,
            w1_qt_shffle,
            w2_qt_shffle,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            topk_ids,
            topk_weights,
            moe_buf,
            a1_scale,
            w1_scale,
            w2_scale,
        ) = FmoeTuner.generate_data(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            blockM,
            device,
        )
        a1_scale_t = a1_scale
        if q_type == QuantType.per_1x128:
            a1_scale_t = a1_scale.t().contiguous()
        ##smooth scale
        # [expert, 1, model_dim]
        fc1_smooth_scale = torch.randn(
            (expert, 1, model_dim), dtype=dtypes.fp32, device=device
        )
        # [expert, 1, inter_dim]
        fc2_smooth_scale = torch.randn(
            (expert, 1, inter_dim), dtype=dtypes.fp32, device=device
        )
        fc1_smooth_scale = None
        fc2_smooth_scale = None
        if q_type == QuantType.per_1x32:
            a1_scale = moe_mxfp4_sort(
                a1_scale,
                sorted_ids,
                num_valid_ids,
                token,
                blockM,
            )
            w1_scale = w1_scale.view(expert, -1)
            w2_scale = w2_scale.view(expert, -1)

        return (
            input,  # 0
            a1_qt,  # 1
            w1_qt_shffle,  # 2
            w2_qt_shffle,  # 3
            sorted_ids,  # 4
            sorted_weights,  # 5
            sorted_expert_ids,  # 6
            num_valid_ids,  # 7
            moe_buf,  # 8
            a1_scale,  # 9
            w1_scale,  # 10
            w2_scale,  # 11
            w1_qt,  # 12
            w2_qt,  # 13
            topk_weights,  # 14
            topk_ids,  # 15
            fc1_smooth_scale,  # 16
            fc2_smooth_scale,  # 17
            a1_scale_t,
        )

    @staticmethod
    def run_torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a1_scale,
        w1_scale,
        sorted_ids=None,
        num_valid_ids=None,
        w1_bias=None,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight_stage1=False,
        topk=1,
        blockM=32,
        fuse_fp4=False,
        fuse_fp8=False,
    ):
        # a16wi4: convert int8 weights to i4x2 so reference function detects the right path
        if (
            quant_type == QuantType.per_1x32
            and w1_qt.dtype == dtypes.i8
            and w1_scale is not None
            and w1_scale.dtype == dtypes.bf16
        ):
            w1_qt = w1_qt.view(dtypes.i4x2)
            w2_qt = w2_qt.view(dtypes.i4x2)
        ref1 = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            activation=activation,
            quant_type=quant_type,
            dtype=dtype,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            w1_bias=w1_bias,
            doweight=doweight_stage1,
        )
        token_num = a1_qt.shape[0]
        if fuse_fp4:
            from aiter.ops.quant import per_1x32_f4_quant

            a2, a2_scale = per_1x32_f4_quant(ref1, quant_dtype=dtypes.fp4x2)
            return a2.view(token_num, topk, -1)
        elif fuse_fp8:
            inter_dim = ref1.shape[-1]
            a2_fp8_bytes, _a2_scale_e8m0 = torch_dynamic_mxfp8_quant(
                ref1.reshape(-1, inter_dim)
            )
            a2 = a2_fp8_bytes.view(dtypes.fp8).view(token_num, topk, inter_dim)
            return a2

        if quant_type == QuantType.per_1x128:
            ref1, ref_scale = aiter.pertoken_quant(
                ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
            )
            ref1 = ref1.view(ref1.shape[0], topk, -1)
        return ref1

    @staticmethod
    def run_torch_moe_stage2(
        a2_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a2_scale,
        w2_scale,
        w2_bias=None,
        dtype=dtypes.bf16,
        quant_type=QuantType.No,
        doweight_stage1=False,
    ):
        # a16wi4: convert int8 weights to i4x2 so reference function detects the right path
        if (
            quant_type == QuantType.per_1x32
            and w2_qt.dtype == dtypes.i8
            and w2_scale is not None
            and w2_scale.dtype == dtypes.bf16
        ):
            w1_qt = w1_qt.view(dtypes.i4x2)
            w2_qt = w2_qt.view(dtypes.i4x2)
        return torch_moe_stage2(
            a2_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype,
            quant_type,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            w2_bias=w2_bias,
            doweight=not doweight_stage1,
        )

    @staticmethod
    def run_torch_moe_stage1_ref(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a1_scale,
        w1_scale,
        dtype,
        activation,
        quant_type,
        doweight_stage1,
        topk,
    ):
        ref1 = FmoeTuner.run_torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            activation=activation,
            quant_type=quant_type,
            dtype=dtype,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            doweight_stage1=doweight_stage1,
            topk=topk,
        )
        token = a1_qt.shape[0]
        inter_dim = w2_qt.shape[-1]
        if quant_type == QuantType.per_1x128:
            ref1, ref_scale = aiter.pertoken_quant(
                ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
            )
            ref1 = ref1.view(ref1.shape[0], topk, -1)
            ref_scale = ref_scale.view(token, -1)
            a2_qt = ref1
            a2_qt = a2_qt.view(token, topk, -1)
            a2_scale = ref_scale
            ratio = a1_scale.element_size() // a1_qt.element_size()
            out1 = torch.zeros(
                (token + (token * ratio + 127) // 128, topk, inter_dim),
                dtype=a1_qt.dtype,
            )
            ref1_asm = torch.zeros_like(out1)
            ref1_asm[:token] = a2_qt
            ref1_asm[token:, ...].view(-1)[
                : token * topk * inter_dim * ratio // 128
            ] = a2_scale.view(a1_qt.dtype).view(-1)
            return ref1_asm

        else:
            out1 = torch.empty(
                (token, topk, inter_dim),
                dtype=dtype,
            )
            return ref1

    ## 1 stage ref
    @staticmethod
    def torch_moe_test(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        # following for int8 quant
        fc1_scale=None,  # [expert, inter_dim, 1]
        fc2_scale=None,  # [expert, model_dim, 1]
        fc1_smooth_scale=None,  # [expert, 1, model_dim]
        fc2_smooth_scale=None,  # [expert, 1, inter_dim]
        activation=ActivationType.Silu,
        doweight_stage1=False,
        q_type_a=dtypes.fp8,
    ):
        if doweight_stage1 & (q_type_a == dtypes.fp8):
            return FmoeTuner.torch_moe_tkw1(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                None,
                activation,
            )
        return torch_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            None,
            activation,
        )

    @staticmethod
    def torch_moe_tkw1(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        # following for int8 quant
        fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
        fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
        fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
        fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
        expert_mask=None,
        activation=ActivationType.Silu,
    ):
        computeType = dtypes.fp32
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(computeType)
        w1 = w1.to(computeType)
        w2 = w2.to(computeType)
        B, D = hidden_states.shape
        topk = topk_weight.shape[1]
        if expert_mask is not None:
            local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
            local_expert_hash[expert_mask == 0] = -1
            topk_ids = local_expert_hash[topk_ids]

        hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
        out = torch.zeros(
            (B, topk, D),
            dtype=computeType,
            device=hidden_states.device,
        )

        inter_dim = w2.shape[2]
        if w2.shape[2] * 2 == w1.shape[1]:
            # g1u1(w1 include gate and up)
            moeType = "g1u1"
        else:
            # g1u0(w1 only include gate)
            moeType = "g1u0"

        if fc1_scale is not None:
            # gose to quant D_w8a8/w8a8
            expert = w1.shape[0]
            w2D = w2.shape[-1]
            w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
            w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

        if fc1_smooth_scale is not None:
            expert = fc1_smooth_scale.shape[0]
            fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
            fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

        for E_id in range(w1.shape[0]):
            mask = topk_ids == E_id
            if mask.sum():
                sub_tokens = hidden_states[mask]
                if fc1_smooth_scale is not None:
                    sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

                act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
                if moeType == "g1u1":
                    gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                    gate = gate * (topk_weight.view(B, -1, 1)[mask])
                    up = up * (topk_weight.view(B, -1, 1)[mask])
                    if activation == ActivationType.Gelu:
                        act_out = F.gelu(gate) * up
                    else:
                        act_out = F.silu(gate) * up
                else:
                    if activation == ActivationType.Gelu:
                        act_out = F.gelu(act_input)
                    else:
                        act_out = F.silu(act_input)
                if fc2_smooth_scale is not None:
                    act_out = act_out * (fc2_smooth_scale[E_id])
                act_out, act_out_scale = aiter.pertoken_quant(
                    act_out, quant_dtype=dtypes.fp8, dtypeMax=None
                )
                out[mask] = (
                    act_out.to(computeType)
                    @ (w2[E_id].transpose(0, 1))
                    * act_out_scale.view(-1, 1)
                )

        return out.sum(dim=1).to(dtype)

    @staticmethod
    def torch_moe_2stages(
        hidden_states,
        w1,  # E, inter_dim*2, model_dim
        w2,  # E, model_dim, inter_dim
        topk_weight,
        topk_ids,
        a1_scale=None,
        w1_scale=None,
        w2_scale=None,
        dtype=dtypes.fp16,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        doweight_stage1=False,
    ):
        ref1 = torch_moe_stage1(
            hidden_states,
            w1,  # E, inter_dim*2, model_dim
            w2,  # E, model_dim, inter_dim
            topk_weight,
            topk_ids,
            dtype=dtype,
            activation=activation,
            quant_type=quant_type,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            doweight=doweight_stage1,
        )
        AQDType = hidden_states.dtype

        if quant_type == aiter.QuantType.per_1x128:
            a2_qt, a2_scale = aiter.pertoken_quant(
                ref1.view(hidden_states.shape[0], -1, 128), quant_dtype=AQDType
            )
        else:
            torch_quant = aiter.get_torch_quant(quant_type)
            a2_qt, a2_scale = torch_quant(ref1, quant_dtype=AQDType)
        a2_qt = a2_qt.view(ref1.shape[0], ref1.shape[1], -1)

        ref2 = torch_moe_stage2(
            a2_qt,
            w1,  # E, inter_dim*2, model_dim
            w2,  # E, model_dim, inter_dim
            topk_weight,
            topk_ids,
            dtype=dtype,
            quant_type=quant_type,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            doweight=not doweight_stage1,
        )
        return ref2

    @staticmethod
    def torch_moe_blockscale(
        hidden_states,
        w1,  # [expert, inter_dim*2, model_dim]
        w2,  # [expert, model_dim, inter_dim]
        topk_weight,
        topk_ids,
        # following for quant
        a_scale=None,
        # [expert, inter_dim/blk_m, model_dim/blk_k]
        fc1_scale=None,
        # [expert, model_dim/blk_m, inter_dim/blk_k]
        fc2_scale=None,
        expert_mask=None,
        scale_blks=(128, 128),
        dtype=dtypes.bf16,
    ):
        computeType = dtypes.fp32
        hidden_states = hidden_states.to(computeType)
        w1 = w1.to(computeType)
        w2 = w2.to(computeType)
        token_num, topk = topk_ids.shape
        expert, model_dim, inter_dim = w2.shape
        B, D = hidden_states.shape
        topk = topk_weight.shape[1]
        if expert_mask is not None:
            local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
            local_expert_hash[expert_mask == 0] = -1
            topk_ids = local_expert_hash[topk_ids]

        blk_n, blk_k = scale_blks
        if a_scale is not None:
            hidden_states = hidden_states.view(
                token_num, -1, blk_k
            ) * a_scale.unsqueeze(-1)
            hidden_states = hidden_states.view(token_num, -1)

        hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
        out = torch.zeros(
            (B, topk, D),
            dtype=computeType,
            device=hidden_states.device,
        )
        if w2.shape[2] * 2 == w1.shape[1]:
            moeType = "g1u1"
        else:
            moeType = "g1u0"

        nblk_n = inter_dim // blk_n
        nblk_k = model_dim // blk_k
        if fc1_scale is not None:
            fc1_scale = rearrange(
                fc1_scale.view(-1, 1)
                .repeat(1, blk_n * blk_k)
                .view(expert, -1, nblk_k, blk_n, blk_k),
                "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
            )
            fc2_scale = rearrange(
                fc2_scale.view(-1, 1)
                .repeat(1, blk_n * blk_k)
                .view(expert, nblk_k, nblk_n, blk_k, blk_n),
                "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
            )
            w1 = w1 * fc1_scale
            w2 = w2 * fc2_scale

        for E_id in range(w1.shape[0]):
            mask = topk_ids == E_id
            if mask.sum():
                sub_tokens = hidden_states[mask]
                act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
                if moeType == "g1u1":
                    gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                    act_out = F.silu(gate) * up
                else:
                    act_out = F.gelu(act_input)
                out[mask] = act_out @ (w2[E_id].transpose(0, 1))

        return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)

    def calculate(self, results, bpes=(1, 1, 2)):
        key, stage, kernelName, block_m, us, err = results
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = key
        if us == self.INVALID_TIME or us == self.INF_TIME:
            return 0, 0
        flop = 0
        data_bytes = 0
        stage = ""
        if stage == "stage1":
            ## gemm1
            # input [token, topk, inter_dim]
            # weight [exprt, 2*inter_dim, model_dim]
            m = token
            k = model_dim
            if use_g1u1:
                n = inter_dim * 2
            else:
                n = inter_dim
            flop = m * n * k * topk * 2
            data_bytes = (
                m * k * self.get_bpe(q_dtype_a)
                + m * n * self.get_bpe(dtype)
                + k * n * self.get_bpe(q_dtype_w) * expert
            )
        elif stage == "stage2":
            ## gemm2
            m = token
            n = model_dim
            k = inter_dim
            b = topk
            # input [token, topk, inter_dim]
            # weight [exprt, dim, inter_dim]
            flop = b * m * n * k * 2
            data_bytes = (
                m * k * self.get_bpe(q_dtype_a) * topk
                + m * n * self.get_bpe(dtype)
                + k * n * self.get_bpe(q_dtype_w) * expert
            )
        else:
            if use_g1u1:
                n = inter_dim * 2
            else:
                n = inter_dim
            flop = (
                token * n * model_dim * topk * 2
                + topk * token * model_dim * inter_dim * 2
            )
            data_bytes = (
                token * model_dim * self.get_bpe(q_dtype_a)
                + n * model_dim * self.get_bpe(q_dtype_w) * expert
                + inter_dim * model_dim * self.get_bpe(q_dtype_w) * expert
                + token * model_dim * self.get_bpe(dtype)
            )  # Rough Estimate
        tflops = round(flop / (us * 1000000), 2)
        bw = round(data_bytes / (us * 1e-6) / 1e9, 2)
        return tflops, bw

    def get_1stage_file_info(self, q_type, q_dtype_a, doweight_stage1):
        if get_gfx() == "gfx950":
            extraInfo_1stage = ""
            if q_dtype_a == dtypes.i8:
                quantDtype = "Int8"
            elif q_dtype_a == dtypes.fp8:
                quantDtype = "Fp8"
            else:
                quantDtype = ""
            if doweight_stage1:
                extraInfo_1stage = "_tkw1"
            if q_type == QuantType.No:
                quantDtype_1stage = "noquant"
            elif q_type == QuantType.per_1x128:
                quantDtype_1stage = "blockscale" + quantDtype
            elif q_type == QuantType.per_1x32:
                quantDtype_1stage = "pertoken" + "MXfp4"
            else:
                quantDtype_1stage = "pertoken" + quantDtype
            return quantDtype_1stage, extraInfo_1stage
        elif get_gfx() == "gfx942":
            extraInfo_1stage = ""
            if q_dtype_a == dtypes.i8:
                quantDtype = "Int8"
            elif q_dtype_a == dtypes.fp8:
                quantDtype = "Fp8"
            else:
                quantDtype = ""
            if doweight_stage1:
                extraInfo_1stage = "_tkw1"
            if q_type == QuantType.No:
                quantDtype_1stage = "noquant"
            elif q_type == QuantType.per_1x128:
                quantDtype_1stage = "blockscale" + quantDtype
            else:
                quantDtype_1stage = "pertoken" + quantDtype
            return quantDtype_1stage, extraInfo_1stage

    def gen_1stage_asm_task(self, key):
        task_1stage = []
        info = key
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info
        ## asm moe 1 stage tuning
        get_gfx()
        key = (act_type, q_type, dtype, q_dtype_a, q_dtype_w, use_g1u1)
        acti_dir = ""
        if act_type == ActivationType.Silu:
            acti_dir = "silu"
        elif act_type == ActivationType.Gelu:
            acti_dir = "gelu"
        up = 1 if use_g1u1 else 0
        quantDtype_1stage, extraInfo_1stage = self.get_1stage_file_info(
            q_type, q_dtype_a, doweight_stage1
        )
        kernels_list_csv_1stage = f"{get_asm_dir()}/fmoe/{acti_dir}/fmoe_bf16_{{quantDtype_1stage}}_g1u{up}_{acti_dir}{{extraInfo_1stage}}.csv"
        asm_kernels_1stage = {}
        if (
            q_type != QuantType.No
            and q_type != QuantType.per_Tensor
            and q_dtype_w != torch.int4
        ):
            asm_kernels_1stage = self.get_kernels_dict(
                kernels_list_csv_1stage.format(
                    quantDtype_1stage=quantDtype_1stage,
                    extraInfo_1stage=extraInfo_1stage,
                ),
                key=["subGU_m", "subGU_n", "smf"],
            )
        fmoe_func = FmoeTuner.get_1stage_fmoe_func(
            q_type, q_dtype_a, act_type, use_g1u1, doweight_stage1
        )
        if fmoe_func is None:
            return task_1stage
        for tile_m, tile_n, smf in asm_kernels_1stage.keys():
            if inter_dim % tile_n != 0 or smf != 0:
                continue

            for el in asm_kernels_1stage.get((tile_m, tile_n, 0), []):
                task_1stage.append(
                    (
                        (info, "asm_1stage", el, tile_m),
                        FmoeTuner.generate_data_1stage,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            tile_m,
                        ),
                        fmoe_func,
                        (
                            [0, 1, 2, 3, 4, 5, 6, 7, 18, 10, 11, 17],
                            q_type,
                            use_g1u1,
                            act_type,
                            el,
                            topk,
                            dtype,
                        ),
                        {},
                        (
                            FmoeTuner.torch_moe_blockscale
                            if q_type == QuantType.per_1x128
                            else FmoeTuner.torch_moe_2stages
                        ),
                        (
                            (
                                [1, 12, 13, 14, 15, 9, 10, 11],
                                None,
                                (128, 128),
                                dtype,
                            )
                            if q_type == QuantType.per_1x128
                            else (
                                [1, 12, 13, 14, 15, 9, 10, 11],
                                dtype,
                                act_type,
                                q_type,
                                doweight_stage1,
                            )
                        ),
                        {},
                        (None),
                        0.01,
                        1,
                        True,
                    )
                )

        return task_1stage

    def gen_2stages_asm1_task(self, key, blockMs):
        info = key
        tasks = []
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info
        kernels_list_csv = f"{get_asm_dir()}/fmoe_2stages/fmoe_stage1_bf16_pertoken{{quantDtype}}{{extraInfo}}_g1u1.csv"
        extraInfo = ""
        if q_type == QuantType.per_1x128:
            extraInfo += "_blockscale"
        if doweight_stage1:
            extraInfo += "_doweight"

        if q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        elif q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        else:
            quantDtype = ""
        asm_kernels = self.get_kernels_dict(
            kernels_list_csv.format(quantDtype=quantDtype, extraInfo=extraInfo)
        )
        for blockM in blockMs:
            # per_1x32 + fp4x2 is a8w4 (MX-FP8 act + MX-FP4 weight); no ASM kernel exists
            # for this combo — the pertokenFp8 CSV only covers per_Token quant.
            if (
                use_g1u1
                and q_dtype_w != torch.int4
                and not (q_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp4x2)
            ):
                for el in asm_kernels.get(blockM, []):
                    tasks.append(
                        (
                            (info, "stage1", el, blockM),  # tag
                            FmoeTuner.generate_asm_stage1,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                            ),
                            FmoeTuner.run_asm_stage1,  # func
                            (
                                [
                                    0,
                                    1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    7,
                                    8,
                                    9,
                                ],  # index of args in generate_asm_stage1
                                topk,
                                blockM,
                                el,
                                0,
                                act_type,
                                q_type,
                                doweight_stage1,
                            ),
                            {},
                            FmoeTuner.run_torch_moe_stage1_ref,
                            (
                                [
                                    0,
                                    12,
                                    13,
                                    10,
                                    11,
                                    14,
                                    9,
                                ],  # index of args in generate_asm_stage1
                                dtype,
                                act_type,
                                q_type,
                                doweight_stage1,
                                topk,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            True,
                        )
                    )
        return tasks

    def gen_2stages_task(self, key, blockMs):
        info = key
        tasks_ck = []
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info

        _is_a8w4 = (
            q_dtype_a == dtypes.fp8
            and q_dtype_w == dtypes.fp4x2
            and q_type == QuantType.per_1x32
        )

        if _is_a8w4:
            return self._gen_2stages_task_cktile(info, blockMs)

        # CK kernels don't support a16wi4 (per_1x32 + i4x2); skip to FlyDSL path
        if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
            return tasks_ck

        _, ck_stage1_kernels = get_gemm1_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,
            int(q_type),
            str(act_type).split(".")[-1].lower(),
            doweight_stage1,
            True,  # bpreshuffle
        )

        is_fp8_blockscale = (
            q_type == QuantType.per_1x128
            and q_dtype_a == dtypes.fp8
            and q_dtype_w == dtypes.fp8
        )
        ck_stage1_splitk_kernels = {}
        splitk_list = []
        if is_fp8_blockscale:
            tilek = 128
            for _sk in range(2, 9):
                if (model_dim % _sk == 0) and ((model_dim // _sk) % tilek == 0):
                    splitk_list.append(_sk)
            if splitk_list:
                _, ck_stage1_splitk_kernels = get_gemm1_kernels_list(
                    dtype2str_dict[q_dtype_a],
                    dtype2str_dict[q_dtype_w],
                    dtype2str_dict[dtype],
                    False,
                    int(q_type),
                    str(act_type).split(".")[-1].lower(),
                    doweight_stage1,
                    True,  # bpreshuffle
                    splitk=True,
                )

        _, ck_stage2_kernels = get_gemm2_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,
            int(q_type),
            not doweight_stage1,
            True,  # bpreshuffle
        )
        for blockM in blockMs:
            if blockM in [16, 32, 64, 128] and use_g1u1:
                for kernel in ck_stage1_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
                    tasks_ck.append(
                        (
                            (info, "stage1", kernel.name, blockM),  # tag
                            FmoeTuner.generate_data_2stages,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                                1,
                            ),
                            FmoeTuner.ck_moe_stage1_fwd_out,  # func
                            (
                                [0, 1, 2, 5, 6, 7, 8, 15, 14],
                                dtype,
                                topk,
                                kernel.name,
                                blockM,
                                q_type,
                                act_type,
                            ),
                            {},
                            FmoeTuner.run_torch_moe_stage1,
                            (
                                [0, 10, 11, 12, 13, 3, 4, 5, 8, 22],
                                dtype,
                                act_type,
                                q_type,
                                doweight_stage1,
                                topk,
                                blockM,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            None,
                        )
                    )

                for kernel in ck_stage2_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
                    s2_ref_args = (
                        [0, 10, 11, 12, 13, 3, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    )
                    tasks_ck.append(
                        (
                            (info, "stage2", kernel.name, blockM),  # tag
                            FmoeTuner.generate_data_2stages,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                                2,
                            ),
                            FmoeTuner.ck_moe_stage2_fwd_out,  # func
                            (
                                [0, 1, 2, 5, 6, 7, 8, 15, 14],
                                dtype,
                                topk,
                                kernel.name,
                                blockM,
                                q_type,
                                act_type,
                            ),
                            {},
                            FmoeTuner.run_torch_moe_stage2,
                            s2_ref_args,
                            {},
                            (None),
                            0.01,
                            0.01,
                            None,
                        )
                    )
        return tasks_ck

    def _gen_2stages_task_cktile(self, info, blockMs):
        """A8W4 (fp8 activation + fp4 weight + per_1x32) uses cktile path."""
        tasks_ck = []
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info

        _gen_data_args_s1 = (
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        )
        _gen_data_args_s2 = (
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        )

        for blockM in blockMs:
            if blockM not in [32, 64] or not use_g1u1:
                continue

            cktile_s1_name = f"cktile_a8w4_bm{blockM}"
            tasks_ck.append(
                (
                    (info, "stage1", cktile_s1_name, blockM),
                    FmoeTuner.generate_data_2stages,
                    (*_gen_data_args_s1, blockM, 1),
                    FmoeTuner.cktile_moe_stage1_out,
                    (
                        [20, 1, 2, 5, 6, 7, 8, 15, 22],
                        dtype,
                        topk,
                        blockM,
                        act_type,
                    ),
                    {},
                    FmoeTuner.run_torch_moe_stage1,
                    (
                        [0, 10, 11, 12, 13, 3, 4, 5, 8, 22],
                        dtype,
                        act_type,
                        q_type,
                        doweight_stage1,
                        topk,
                        blockM,
                    ),
                    {},
                    (None),
                    0.01,
                    0.01,
                    cosine_diff_compare,
                )
            )

            cktile_s2_name = f"cktile_a8w4_bm{blockM}"
            tasks_ck.append(
                (
                    (info, "stage2", cktile_s2_name, blockM),
                    FmoeTuner.generate_data_2stages,
                    (*_gen_data_args_s2, blockM, 2),
                    FmoeTuner.cktile_moe_stage2_out,
                    (
                        [0, 1, 2, 5, 6, 7, 8, 15, 14, 22],
                        dtype,
                        topk,
                        blockM,
                        act_type,
                    ),
                    {},
                    FmoeTuner.run_torch_moe_stage2,
                    (
                        [20, 10, 11, 12, 13, 21, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    ),
                    {},
                    (None),
                    0.01,
                    0.01,
                    cosine_diff_compare,
                )
            )

        return tasks_ck

    def gen_flydsl_2stages_task(self, info, blockMs):
        tasks_flydsl = []
        if not is_flydsl_available():
            return tasks_flydsl
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info

        if q_type != QuantType.per_1x32 or q_dtype_w != dtypes.fp4x2:
            return tasks_flydsl

        _a_dtype_map = {
            dtypes.fp8: "fp8",
            dtypes.fp4x2: "fp4",
            dtypes.fp16: "fp16",
            dtypes.bf16: "fp16",
        }
        a_dtype_str = _a_dtype_map.get(q_dtype_a, "fp8")
        b_dtype_str = "fp4"
        out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"

        flydsl_s1_kernels = get_flydsl_stage1_kernels(
            a_dtype_str, b_dtype_str, out_dtype_str
        )
        flydsl_s2_kernels = get_flydsl_stage2_kernels(
            a_dtype_str, b_dtype_str, out_dtype_str
        )

        for blockM in blockMs:
            if blockM not in [32, 64, 128] or not use_g1u1:
                continue
            for kname, kparams in flydsl_s1_kernels.items():
                is_splitk = kparams.get("k_batch", 1) > 1

                # (kernel_name, kparams, is_fp4, is_fp8)
                # out_dtype encodes fused quant type: "fp4" or "fp8"
                #   a8w4 (a_dtype_str="fp8"): stage2 expects fp8 activations → out_dtype="fp8"
                #   a4w4 (a_dtype_str="fp4"): stage2 expects fp4 activations → out_dtype="fp4"
                s1_tile_m = kparams["tile_m"]
                if s1_tile_m != blockM:
                    continue
                if a_dtype_str == "fp8":
                    fp8_params = {
                        **kparams,
                        "out_dtype": "fp8",
                        "a_scale_one": True,
                        "gate_mode": "interleave",
                    }
                    nonfused_params = {**kparams, "a_scale_one": True}
                    if is_splitk:
                        s1_variants = [(kname + "_fp8", fp8_params, False, True)]
                    else:
                        s1_variants = [
                            (kname, nonfused_params, False, False),
                            (kname + "_fp8", fp8_params, False, True),
                        ]
                else:
                    fp4_params = {**kparams, "out_dtype": "fp4"}
                    if is_splitk:
                        s1_variants = [(kname + "_fp4", fp4_params, True, False)]
                    else:
                        s1_variants = [
                            (kname, kparams, False, False),
                            (kname + "_fp4", fp4_params, True, False),
                        ]

                for s1_name, s1_params, is_fp4, is_fp8 in s1_variants:
                    s1_compare_fn = None
                    if is_fp8 or a_dtype_str == "fp8":
                        # a8w4: precision differs from torch ref; use cosine
                        # diff (logits diff) instead of checkAllclose.
                        s1_compare_fn = cosine_diff_compare
                    ref_args_extra = (
                        [0, 10, 11, 12, 13, 3, 4, 5, 8, 22],
                        dtype,
                        act_type,
                        q_type,
                        doweight_stage1,
                        topk,
                        blockM,
                    )
                    if is_fp4:
                        ref_args_extra = ref_args_extra + (True,)
                    elif is_fp8:
                        ref_args_extra = ref_args_extra + (False, True)
                    s1_ref_func = FmoeTuner.run_torch_moe_stage1
                    s1_ref_args = ref_args_extra
                    s1_ref_kwargs = {}
                    s1_ref = None

                    # _fp8 variant uses direct fp8-cast activation (index 20)
                    a1_idx = 20 if is_fp8 else 0
                    tasks_flydsl.append(
                        (
                            (info, "stage1", s1_name, blockM),
                            FmoeTuner.generate_data_2stages,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                                1,
                            ),
                            FmoeTuner.run_flydsl_stage1_out,
                            (
                                [a1_idx, 1, 5, 6, 7, 8, 15, 14, 22],
                                dtype,
                                topk,
                                s1_params,
                                blockM,
                                q_dtype_a,
                                q_type,
                                act_type,
                            ),
                            {},
                            s1_ref_func,
                            s1_ref_args,
                            s1_ref_kwargs,
                            s1_ref,
                            0.01,
                            0.01,
                            s1_compare_fn,
                        )
                    )

            for kname, kparams in flydsl_s2_kernels.items():
                s2_tile_m = kparams["tile_m"]
                if blockM % s2_tile_m != 0:
                    continue
                # Only try matched (tile_m==blockM) and one smaller (blockM/2) to limit candidates
                if s2_tile_m != blockM and s2_tile_m != blockM // 2:
                    continue
                s2_kparams = {**kparams, "sort_block_m": blockM}
                s2_kname = kname if s2_tile_m == blockM else f"{kname}_sbm{blockM}"

                s2_ref_kwargs = {}
                s2_compare_fn = None
                if a_dtype_str == "fp8":
                    s2_compare_fn = cosine_diff_compare
                    # Use bf16 stage1 output (idx 20) and a2_scale=None (idx 21)
                    # so torch ref takes the a16w4 path instead of mxfp4_to_f32.
                    s2_ref_args = (
                        [20, 10, 11, 12, 13, 21, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    )
                else:
                    s2_ref_args = (
                        [0, 10, 11, 12, 13, 3, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    )
                s2_ref_func = FmoeTuner.run_torch_moe_stage2

                tasks_flydsl.append(
                    (
                        (info, "stage2", s2_kname, blockM),
                        FmoeTuner.generate_data_2stages,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            doweight_stage1,
                            blockM,
                            2,
                        ),
                        FmoeTuner.run_flydsl_stage2_out,
                        (
                            [0, 17, 5, 6, 7, 8, 19, 14, 9, 22],
                            dtype,
                            topk,
                            s2_kparams,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {},
                        s2_ref_func,
                        s2_ref_args,
                        s2_ref_kwargs,
                        (None),
                        0.01,
                        0.01,
                        s2_compare_fn,
                    )
                )

        return tasks_flydsl

    def gen_flydsl_i4_2stages_task(self, info, blockMs):
        tasks_flydsl = []
        if not is_flydsl_available():
            return tasks_flydsl
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = info

        if not (q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2):
            return tasks_flydsl

        out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"
        _a_dtype_map = {
            dtypes.fp8: "fp8",
            dtypes.fp4x2: "fp4",
            dtypes.fp16: "fp16",
            dtypes.bf16: "fp16",
        }
        a_dtype_str = _a_dtype_map.get(q_dtype_a, "fp8")

        flydsl_s1_kernels = get_flydsl_stage1_kernels_int4_bf16(out_dtype_str)
        flydsl_s2_kernels = get_flydsl_stage2_kernels_int4_bf16(out_dtype_str)

        for blockM in blockMs:
            if blockM not in [16, 32, 64, 128] or not use_g1u1:
                continue
            for kname, kparams in flydsl_s1_kernels.items():
                ktm = kparams["tile_m"]
                if ktm != blockM:
                    continue
                # Validate split-k compatibility with model_dim
                kb = kparams.get("k_batch", 1)
                if kb > 1:
                    if model_dim % kb != 0:
                        continue
                    k_per_batch = model_dim // kb
                    if k_per_batch % kparams["tile_k"] != 0:
                        continue
                    k_tiles = k_per_batch // kparams["tile_k"]
                    if k_tiles < 4 or k_tiles % 2 != 0:
                        continue
                # Int4 kernels: no fuse_fp4_quant
                ref_args_extra = (
                    [0, 10, 11, 12, 13, 3, 4, 5, 8, 22],
                    dtype,
                    act_type,
                    q_type,
                    doweight_stage1,
                    topk,
                    blockM,
                )
                tasks_flydsl.append(
                    (
                        (info, "stage1", kname, blockM),
                        FmoeTuner.generate_data_2stages,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            doweight_stage1,
                            blockM,
                            1,
                        ),
                        FmoeTuner.run_flydsl_stage1_out,
                        (
                            [0, 16, 5, 6, 7, 8, 18, 14, 22],
                            dtype,
                            topk,
                            kparams,
                            blockM,
                            q_dtype_a,
                            q_type,
                            act_type,
                        ),
                        {},
                        FmoeTuner.run_torch_moe_stage1,
                        ref_args_extra,
                        {},
                        (None),
                        0.01,
                        0.01,
                        True,
                    )
                )

            for kname, kparams in flydsl_s2_kernels.items():
                s2_tile_m = kparams["tile_m"]
                if s2_tile_m != blockM:
                    continue
                if kparams.get("mode", "atomic") != "atomic":
                    continue
                if kparams.get("persist", None) is not None:
                    continue
                s2_kparams = {**kparams, "sort_block_m": blockM}
                s2_kname = kname if s2_tile_m == blockM else f"{kname}_sbm{blockM}"

                s2_ref_kwargs = {}
                s2_compare_fn = None
                if a_dtype_str == "fp8":
                    s2_compare_fn = cosine_diff_compare
                    # Use bf16 stage1 output (idx 20) and a2_scale=None (idx 21)
                    # so torch ref takes the a16w4 path instead of mxfp4_to_f32.
                    s2_ref_args = (
                        [20, 10, 11, 12, 13, 21, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    )
                else:
                    s2_ref_args = (
                        [0, 10, 11, 12, 13, 3, 4, 22],
                        dtype,
                        q_type,
                        doweight_stage1,
                    )
                s2_ref_func = FmoeTuner.run_torch_moe_stage2

                tasks_flydsl.append(
                    (
                        (info, "stage2", s2_kname, blockM),
                        FmoeTuner.generate_data_2stages,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            doweight_stage1,
                            blockM,
                            2,
                        ),
                        FmoeTuner.run_flydsl_stage2_out,
                        (
                            [0, 17, 5, 6, 7, 8, 19, 14, 9, 22],
                            dtype,
                            topk,
                            s2_kparams,
                            blockM,
                            q_type,
                            act_type,
                        ),
                        {},
                        s2_ref_func,
                        s2_ref_args,
                        s2_ref_kwargs,
                        (None),
                        0.01,
                        0.01,
                        s2_compare_fn,
                    )
                )

        return tasks_flydsl

    def run_config(self, args):
        from aiter.fused_moe import fused_moe, fused_topk
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            token = int(row["token"])
            model_dim = int(row["model_dim"])
            inter_dim = int(row["inter_dim"])
            expert = int(row["expert"])
            topk = int(row["topk"])
            act_type = eval(row["act_type"])
            dtype = eval(row["dtype"])
            q_dtype_a = eval(row["q_dtype_a"])
            q_dtype_w = eval(row["q_dtype_w"])
            q_type = eval(row["q_type"])
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            use_g1u1 = bool(row["use_g1u1"])
            doweight_stage1 = bool(row["doweight_stage1"])
            shape_str = (
                f"({token}, {model_dim}, {inter_dim}, E={expert}, topk={topk}, "
                f"{row['act_type']}, {row['dtype']}, {row['q_dtype_a']}, "
                f"{row['q_dtype_w']}, {row['q_type']}, g1u1={use_g1u1}, "
                f"dw_s1={doweight_stage1})"
            )
            kernel_us = None
            if "us" in row and pd.notna(row["us"]):
                try:
                    kernel_us = float(row["us"])
                except (TypeError, ValueError):
                    kernel_us = None
            try:
                torch.manual_seed(0)
                hidden = (
                    torch.randn((token, model_dim), dtype=dtype, device="cuda") / 10
                )
                if use_g1u1:
                    w1 = (
                        torch.randn(
                            (expert, inter_dim * 2, model_dim),
                            dtype=dtype,
                            device="cuda",
                        )
                        / 10
                    )
                else:
                    w1 = (
                        torch.randn(
                            (expert, inter_dim, model_dim), dtype=dtype, device="cuda"
                        )
                        / 10
                    )
                w2 = torch.randn(
                    (expert, model_dim, inter_dim), dtype=dtype, device="cuda"
                )
                w1_qt, w1_scale = self.weight_quant(w1, q_type, quant_dtype=q_dtype_w)
                w2_qt, w2_scale = self.weight_quant(w2, q_type, quant_dtype=q_dtype_w)
                if q_dtype_w is not dtypes.fp4x2:
                    w1_qt = w1_qt.view(w1.shape)
                    w2_qt = w2_qt.view(w2.shape)
                else:
                    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
                    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

                # Match the production/test path used by op_tests/test_moe_2stage.py.
                w1_qt_fmoe = w1_qt
                w2_qt_fmoe = w2_qt
                w1_scale_fmoe = w1_scale
                w2_scale_fmoe = w2_scale
                if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
                    w1_qt_fmoe = (
                        pack_int8_to_packed_int4(shuffle_weight(w1_qt_fmoe, (16, 16)))
                        .view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
                        .view(dtypes.i4x2)
                    )
                    w2_qt_fmoe = (
                        pack_int8_to_packed_int4(shuffle_weight(w2_qt_fmoe, (16, 16)))
                        .view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
                        .view(dtypes.i4x2)
                    )
                    w1_scale_fmoe = (
                        shuffle_scale_for_int4(w1_scale, group_size=32)
                        .view(-1)
                        .contiguous()
                    )
                    w2_scale_fmoe = (
                        shuffle_scale_for_int4(w2_scale, group_size=32)
                        .view(-1)
                        .contiguous()
                    )
                elif q_dtype_w == torch.int4:
                    w1_qt_fmoe = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w1_qt_fmoe, (16, 16), use_int4=True)
                        )
                    )
                    w2_qt_fmoe = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w2_qt_fmoe, (16, 16), use_int4=True)
                        )
                    )
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a in [dtypes.bf16, dtypes.fp16, dtypes.fp8]
                    and q_dtype_w == dtypes.fp4x2
                ):
                    w1_qt_fmoe = shuffle_weight_a16w4(w1_qt_fmoe, 16, True)
                    w1_scale_fmoe = shuffle_scale_a16w4(w1_scale, expert, True)
                    w2_qt_fmoe = shuffle_weight_a16w4(w2_qt_fmoe, 16, False)
                    w2_scale_fmoe = shuffle_scale_a16w4(w2_scale, expert, False)
                elif q_dtype_w != dtypes.fp4x2:
                    w1_qt_fmoe = shuffle_weight(w1_qt_fmoe, (16, 16))
                    w2_qt_fmoe = shuffle_weight(w2_qt_fmoe, (16, 16))
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )
                else:
                    w1_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w1_scale)
                        if w1_scale is not None
                        else None
                    )
                    w2_scale_fmoe = (
                        fp4_utils.e8m0_shuffle(w2_scale)
                        if w2_scale is not None
                        else None
                    )

                w1_qt_fmoe.is_shuffled = True
                w2_qt_fmoe.is_shuffled = True

                score = torch.randn((token, expert), dtype=dtype, device="cuda")
                topk_weights, topk_ids = fused_topk(hidden, score, topk, True)
                if q_type == QuantType.per_1x128:
                    a1_qt, a1_scale = aiter.pertoken_quant(
                        hidden.view(token, -1, 128), quant_dtype=q_dtype_a
                    )
                    a1_qt = a1_qt.view(token, model_dim)
                    a1_scale = a1_scale.squeeze(-1)
                elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
                    a1_qt = hidden.to(dtypes.bf16)
                    a1_scale = None
                elif (
                    q_type == QuantType.per_1x32
                    and q_dtype_a in [dtypes.bf16, dtypes.fp16]
                    and q_dtype_w == dtypes.fp4x2
                ):
                    a1_qt = hidden.to(dtype)
                    a1_scale = None
                else:
                    torch_quant = aiter.get_torch_quant(q_type)
                    a1_qt, a1_scale = torch_quant(hidden, quant_dtype=q_dtype_a)

                out, us = run_perftest(
                    fused_moe,
                    hidden,
                    w1_qt_fmoe,
                    w2_qt_fmoe,
                    topk_weights,
                    topk_ids,
                    activation=act_type,
                    quant_type=q_type,
                    doweight_stage1=doweight_stage1,
                    w1_scale=w1_scale_fmoe,
                    w2_scale=w2_scale_fmoe,
                    dtype=dtype,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = self.torch_moe_2stages(
                    a1_qt,
                    w1_qt,
                    w2_qt,
                    topk_weights,
                    topk_ids,
                    a1_scale=a1_scale,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    dtype=dtype,
                    activation=act_type,
                    quant_type=q_type,
                    doweight_stage1=doweight_stage1,
                )
                if out.count_nonzero() == 0 and ref.count_nonzero() > 0:
                    status = "error:output is all zeros (kernel produced no output)"
                    err_ratio = 1.0
                else:
                    err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                    if err_ratio <= args.errRatio:
                        status = "ok"
                    else:
                        status = f"mismatch:err_ratio={err_ratio:.4f}(>{args.errRatio})"
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": us,
                        "kernel_us": kernel_us,
                        "status": status,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "shape": shape_str,
                        "e2e_us": -1,
                        "kernel_us": kernel_us,
                        "status": f"error:{e}",
                    }
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        self._flydsl_fallbacks = []
        mp_num = args.mp
        blockMs = [16, 32, 64, 128]
        keys = self.keys
        tasks = []
        tasks_ck = []
        task_1stage = []
        in_data = []
        for line in untunedf[keys].values:
            (
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            ) = line
            dtype = eval(dtype)
            q_dtype_a = eval(q_dtype_a)
            q_dtype_w = eval(q_dtype_w)
            q_type = eval(q_type)
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            print("\nStart tuning", line)
            if get_gfx() not in ["gfx950"] and q_type in [aiter.QuantType.per_1x32]:
                print(f"{q_type} is not supported on {get_gfx()}")
                return []
            if not use_g1u1:
                print("no moe solution(g1u0) can tune for ", line)
                continue
            act_type = eval(act_type)
            info = (
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            )
            tasks.extend(self.gen_2stages_asm1_task(info, blockMs))
            tasks_ck.extend(self.gen_2stages_task(info, blockMs))
            tasks_ck.extend(self.gen_flydsl_2stages_task(info, blockMs))
            tasks_ck.extend(self.gen_flydsl_i4_2stages_task(info, blockMs))
            task_1stage.extend(self.gen_1stage_asm_task(info))
            if tasks is None and tasks_ck is None and task_1stage is None:
                print("no moe solution can tune for ", line)
                continue
            print(
                f"stage1 asm tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}, task_1stage is {len(task_1stage)}"
            )
        all_tasks = tasks + tasks_ck + task_1stage
        # Record dispatched cases
        dispatched = {}
        for i, task in enumerate(all_tasks):
            tag = task[0]  # (info, stage, kname, blockM)
            dispatched[i] = tag
            if args.verbose:
                _, stage, kname, blockM = tag
                print(f"  [dispatch] task {i}: {stage} {kname} blockM={blockM}")

        in_data.append((len(all_tasks), ()))
        rets = []
        if len(all_tasks) > 0:
            ### shape_grouped should be False as multiple stages
            rets = mp_tuner(
                all_tasks,
                in_data,
                mp_num,
                True,
                False,
                timeout=args.timeout,
                verbose=args.verbose,
            )

        # Identify failed cases
        if rets:
            failed_cases = []
            for i, ret in enumerate(rets):
                info, us, err = ret
                if us == float("inf") or us == -1:
                    tag = dispatched.get(i, info)
                    failed_cases.append((i, tag, us, err))
            if failed_cases:
                print(f"\n[tune] {len(failed_cases)} of {len(rets)} tasks failed:")
                for i, tag, us, err in failed_cases:
                    _, stage, kname, blockM = tag
                    reason = "timeout/hang" if us == float("inf") else "crash/error"
                    print(f"  task {i}: {stage} {kname} blockM={blockM} -> {reason}")
            else:
                print(f"\n[tune] all {len(rets)} tasks completed successfully")

        if not rets:
            print("no shape to tune or no solution found")
            return []
        else:
            return rets

    def result_to_csv(self, results, file, concat=False):
        old_tunedf = self.get_tuned_gemm_list(file)

        new_fallbacks = getattr(self, "_flydsl_fallbacks", [])
        new_fb_keys = set()
        if new_fallbacks:
            new_fb_df = pd.DataFrame(new_fallbacks, columns=self.columns)
            new_fb_keys = set(new_fb_df[self.keys].astype(str).apply(tuple, axis=1))

        if "_tag" in old_tunedf.columns:
            old_fb_mask = old_tunedf["_tag"].fillna("") == FLYDSL_FALLBACK_TAG
            if new_fb_keys:
                old_fb_key_tuples = (
                    old_tunedf.loc[old_fb_mask, self.keys]
                    .astype(str)
                    .apply(tuple, axis=1)
                )
                drop_mask = old_fb_mask & old_fb_key_tuples.isin(new_fb_keys)
            else:
                drop_mask = old_fb_mask & False
            kept_old_fb = old_tunedf[old_fb_mask & ~drop_mask].copy()
            old_tunedf = old_tunedf[~old_fb_mask].drop(columns=["_tag"])
        else:
            kept_old_fb = pd.DataFrame(columns=list(old_tunedf.columns) + ["_tag"])

        resultdf = self.update_tunedf(old_tunedf, results)
        self.success = pd.concat([self.success, results], ignore_index=True)
        resultdf["run_1stage"] = resultdf["run_1stage"].astype(int)
        if results is not None:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys,
                keep="last",
            )
        resultdf["_tag"] = ""

        if new_fallbacks:
            new_fb_df = pd.DataFrame(new_fallbacks, columns=self.columns)
            new_fb_df["_tag"] = FLYDSL_FALLBACK_TAG
            resultdf = pd.concat([resultdf, new_fb_df], ignore_index=True)

        if len(kept_old_fb) > 0:
            resultdf = pd.concat([resultdf, kept_old_fb], ignore_index=True)

        resultdf = resultdf.astype(str).drop_duplicates(
            subset=self.keys + ["_tag"], keep="last"
        )
        resultdf.to_csv(file, index=False)

    def post_process(self, results, args, topk=-1, fast_mode=False):
        profileDF = []
        profileDF = []
        prorfiles = []
        bests = []
        from collections import defaultdict

        ##group results by info[0](key)
        grouped_rets = defaultdict(list)
        for info, us, max_err_ratio in results:
            grouped_rets[tuple(info[0])].append((info[1:], us, max_err_ratio))
        grouped_results = grouped_rets.items()
        for key, rets in grouped_results:
            us_qs_cache = {}
            (
                cu_num,
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                act_type,
                dtype,
                q_dtype_a,
                q_dtype_w,
                q_type,
                use_g1u1,
                doweight_stage1,
            ) = key
            import re

            profileDF = []
            for (stage, kernelName, block_m), us, err in rets:
                tflops, bw = self.calculate((key, stage, kernelName, block_m, us, err))
                row_ksplit = 0
                sk_match = re.search(r"_sk(\d+)$", str(kernelName))
                if sk_match:
                    row_ksplit = int(sk_match.group(1))
                    kernelName = re.sub(r"_sk\d+$", "", kernelName)
                profileDF.append(
                    [
                        stage,
                        cu_num,
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        block_m,
                        row_ksplit,
                        us,
                        kernelName,
                        err,
                        tflops,
                        bw,
                    ]
                )

            profileDF = pd.DataFrame(
                profileDF,
                columns=["stage"]
                # + ["cu_num"]
                + self.keys
                + ["block_m", "ksplit", "us", "kernelName", "err", "tflops", "bw"],
            )
            prorfiles.append(profileDF)

            ## remove invalid candidate
            profileDF = profileDF[
                (profileDF["us"] != float("-inf"))
                & (profileDF["us"] != -1)
                & (profileDF["err"] <= args.errRatio)
            ]
            # Keep best non-flydsl per (stage, block_m) for fallback before dedup
            _non_flydsl = profileDF[
                ~profileDF["kernelName"].astype(str).str.startswith("flydsl_")
            ]
            _non_flydsl_best = _non_flydsl.sort_values("us").drop_duplicates(
                ["stage", "block_m"], keep="first"
            )
            profileDF = profileDF.sort_values("us").drop_duplicates(
                ["stage", "block_m"], keep="first"
            )
            stage1_profileDF = profileDF[profileDF["stage"] == "stage1"].drop(
                columns=["stage"]
            )

            stage1_profileDF = stage1_profileDF.rename(
                columns={
                    "kernelName": "kernelName1",
                    "err": "err1",
                    "us": "us1",
                    "tflops": "tflops1",
                    "bw": "bw1",
                }
            )
            stage2_profileDF = profileDF[profileDF["stage"] == "stage2"].drop(
                columns=["stage", "ksplit"]
            )
            stage2_profileDF = stage2_profileDF.rename(
                columns={
                    "kernelName": "kernelName2",
                    "err": "err2",
                    "us": "us2",
                    "tflops": "tflops2",
                    "bw": "bw2",
                }
            )
            if (stage1_profileDF.shape[0] == 0 and stage2_profileDF.shape[0] != 0) or (
                stage1_profileDF.shape[0] != 0 and stage2_profileDF.shape[0] == 0
            ):
                print(
                    "Error: please check errRatio, stage1 and stage2 should be valid together!"
                )
            asm_1stage_profileDF = profileDF[profileDF["stage"] == "asm_1stage"].drop(
                columns=["stage"]
            )
            asm_1stage_profileDF = asm_1stage_profileDF.rename(
                columns={
                    "kernelName": "kernelName1",
                    "err": "err1",
                    "us": "us1",
                    "tflops": "tflops1",
                    "bw": "bw1",
                }
            )
            empty_1stage_profileDF = pd.DataFrame(index=asm_1stage_profileDF.index)

            empty_1stage_profileDF["kernelName2"] = None
            empty_1stage_profileDF["err2"] = 0
            empty_1stage_profileDF["us2"] = 0
            empty_1stage_profileDF["tflops2"] = 0
            empty_1stage_profileDF["bw2"] = 0
            asm_1stage_profileDF = pd.concat(
                [asm_1stage_profileDF, empty_1stage_profileDF], axis=1
            )
            asm_1stage_profileDF["run_1stage"] = 1
            profileDF = pd.merge(
                stage1_profileDF,
                stage2_profileDF,
                on=[
                    "cu_num",
                    "token",
                    "model_dim",
                    "inter_dim",
                    "expert",
                    "topk",
                    "act_type",
                    "dtype",
                    "q_dtype_a",
                    "q_dtype_w",
                    "q_type",
                    "use_g1u1",
                    "doweight_stage1",
                    "block_m",
                ],
                how="inner",
            )
            profileDF["run_1stage"] = 0
            profileDF = pd.concat([profileDF, asm_1stage_profileDF], axis=0)
            if len(profileDF) == 0:
                print(
                    f"no valid candidate found for {key}, please check the time or errRatio in all result file running with --profile_file"
                )
                ret = []
                ret.append(
                    [
                        cu_num,
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        0,
                        0,
                        self.INVALID_TIME,
                        None,
                        1,
                        self.INVALID_TIME,
                        None,
                        1,
                        self.INVALID_TIME,
                        0,
                        -1,
                        -1,
                    ]
                )
                failedf = pd.DataFrame(ret, columns=self.columns)
                self.failed = pd.concat([self.failed, failedf], axis=0)
                continue
            if q_type == QuantType.per_1x32 and q_dtype_w != dtypes.i4x2:
                # For a4w4 (fp4 activation), a separate fp4-quant+sort step is needed
                # between stage1 (bf16 output) and stage2 (fp4 input).  Benchmark its
                # cost and add it to non-fused kernels so comparisons are fair.
                #
                # For a8w4 (fp8 activation), non-fused paths assume bf16 stage1 output
                # then a separate cast to fp8 before stage2; benchmark that cast
                # (simple .to(dtypes.fp8)) and add it to kernels whose stage1 name does
                # not end with _fp8 (those fuse the cast in stage1).
                if q_dtype_a == dtypes.fp4x2:
                    from aiter.test_common import run_perftest
                    from aiter.ops.triton.quant.fused_mxfp4_quant import (
                        fused_dynamic_mxfp4_quant_moe_sort,
                    )

                    us_qs_cache = {}
                    for bm in profileDF["block_m"].unique():
                        bm_int = int(bm)
                        block_size = max(32, bm_int)
                        num_sorted = (
                            (token * topk + block_size - 1) // block_size
                        ) * block_size
                        dummy_act = torch.randn(
                            token * topk, inter_dim, dtype=dtype, device="cuda"
                        )
                        dummy_sorted_ids = torch.arange(
                            num_sorted, dtype=torch.int32, device="cuda"
                        )
                        dummy_num_valid = torch.tensor(
                            [token * topk], dtype=torch.int32, device="cuda"
                        )
                        _, us_qs = run_perftest(
                            fused_dynamic_mxfp4_quant_moe_sort,
                            dummy_act,
                            sorted_ids=dummy_sorted_ids,
                            num_valid_ids=dummy_num_valid,
                            token_num=token,
                            topk=topk,
                            block_size=block_size,
                        )
                        us_qs_cache[bm] = round(us_qs, 4)
                        print(
                            f"  quant_sort benchmark: blockM={bm_int}, us={us_qs_cache[bm]}"
                        )
                    profileDF["us_quant_sort"] = profileDF["block_m"].map(us_qs_cache)
                    # _fp4 kernels already fuse the fp4-quant+sort; skip cost addition
                    is_fp4 = profileDF["kernelName1"].astype(str).str.endswith("_fp4")
                    profileDF.loc[~is_fp4, "us1"] = (
                        profileDF.loc[~is_fp4, "us1"]
                        + profileDF.loc[~is_fp4, "us_quant_sort"]
                    )
                    profileDF.drop(columns=["us_quant_sort"], inplace=True)
                elif q_dtype_a == dtypes.fp8:
                    from aiter.test_common import run_perftest

                    dummy_act = torch.randn(
                        token * topk, inter_dim, dtype=dtype, device="cuda"
                    )

                    def _act_to_fp8(x):
                        _scale_tmp = torch.ones(
                            [x.shape[0], x.shape[1] // 32],
                            dtype=dtypes.fp8_e8m0,
                            device=x.device,
                        )
                        return x.to(dtypes.fp8)

                    _, us_fp8_cast = run_perftest(_act_to_fp8, dummy_act)
                    us_fp8_cast = round(us_fp8_cast, 4)
                    print(f"  fp8 activation cast benchmark: us={us_fp8_cast}")
                    us_qs_cache = {}
                    for bm in profileDF["block_m"].unique():
                        us_qs_cache[bm] = us_fp8_cast
                    profileDF["us_quant_sort"] = profileDF["block_m"].map(us_qs_cache)
                    is_fp8 = profileDF["kernelName1"].astype(str).str.endswith("_fp8")
                    profileDF.loc[~is_fp8, "us1"] = (
                        profileDF.loc[~is_fp8, "us1"]
                        + profileDF.loc[~is_fp8, "us_quant_sort"]
                    )
                    profileDF.drop(columns=["us_quant_sort"], inplace=True)

            profileDF["us"] = round(profileDF["us1"] + profileDF["us2"], 4)
            results = profileDF.apply(
                lambda row: self.calculate(
                    (
                        tuple(row[col] for col in self.keys),
                        "",
                        row["kernelName1"],
                        row["block_m"],
                        row["us"],
                        row["err1"],
                    )
                ),
                axis=1,
                result_type="expand",
            )
            profileDF["tflops"] = results[0]
            profileDF["bw"] = results[1]
            profileDF.drop(["tflops1", "tflops2", "bw1", "bw2"], axis=1, inplace=True)
            profileDF["err1"] = profileDF["err1"].apply(lambda x: f"{x:.1%}")
            profileDF["err2"] = profileDF["err2"].apply(lambda x: f"{x:.1%}")
            if args.profile_file != "":
                if os.path.exists(args.profile_file):
                    old_df = pd.read_csv(args.profile_file)
                else:
                    old_df = pd.DataFrame(columns=self.columns)
                tmpprofileDF = pd.concat([old_df, profileDF], ignore_index=True)
                tmpprofileDF.to_csv(args.profile_file, index=False)
            best_one = profileDF.loc[profileDF["us"].idxmin()].copy()
            print(
                f"Tuning result for {key} is {best_one['block_m'], best_one['kernelName1'], best_one['kernelName2'], best_one['err1'], best_one['err2'], best_one['run_1stage']} {best_one['us']} us, {best_one['tflops']} TFLOPS, {best_one['bw']} GB/s"
            )
            best_one["act_type"] = str(best_one["act_type"])
            best_one["q_type"] = str(best_one["q_type"])
            best_one["dtype"] = str(best_one["dtype"])
            best_one["q_dtype_a"] = str(best_one["q_dtype_a"])
            best_one["q_dtype_w"] = str(best_one["q_dtype_w"])
            bests.append(best_one)

            best_has_flydsl = str(best_one.get("kernelName1", "")).startswith(
                "flydsl_"
            ) or str(best_one.get("kernelName2", "")).startswith("flydsl_")
            if best_has_flydsl:
                # Build fallback from best non-flydsl candidates (saved before dedup)
                _nf_s1 = (
                    _non_flydsl_best[_non_flydsl_best["stage"] == "stage1"]
                    .drop(columns=["stage"])
                    .rename(
                        columns={
                            "kernelName": "kernelName1",
                            "err": "err1",
                            "us": "us1",
                            "tflops": "tflops1",
                            "bw": "bw1",
                        }
                    )
                )
                _nf_s2 = (
                    _non_flydsl_best[_non_flydsl_best["stage"] == "stage2"]
                    .drop(columns=["stage", "ksplit"])
                    .rename(
                        columns={
                            "kernelName": "kernelName2",
                            "err": "err2",
                            "us": "us2",
                            "tflops": "tflops2",
                            "bw": "bw2",
                        }
                    )
                )
                _join_keys = [
                    c for c in self.keys if c in _nf_s1.columns and c in _nf_s2.columns
                ] + ["block_m"]
                non_flydsl_df = pd.merge(_nf_s1, _nf_s2, on=_join_keys, how="inner")
                if len(non_flydsl_df) > 0:
                    if q_type == QuantType.per_1x32 and us_qs_cache:
                        non_flydsl_df["us_quant_sort"] = non_flydsl_df["block_m"].map(
                            us_qs_cache
                        )
                        non_flydsl_df["us1"] = non_flydsl_df["us1"] + non_flydsl_df[
                            "us_quant_sort"
                        ].fillna(0)
                        non_flydsl_df.drop(columns=["us_quant_sort"], inplace=True)
                    non_flydsl_df["us"] = round(
                        non_flydsl_df["us1"] + non_flydsl_df["us2"], 4
                    )
                    non_flydsl_df["run_1stage"] = 0
                    non_flydsl_df["tflops"] = 0
                    non_flydsl_df["bw"] = 0
                    fb = non_flydsl_df.loc[non_flydsl_df["us"].idxmin()].copy()
                    fb["act_type"] = str(fb["act_type"])
                    fb["q_type"] = str(fb["q_type"])
                    fb["dtype"] = str(fb["dtype"])
                    fb["q_dtype_a"] = str(fb["q_dtype_a"])
                    fb["q_dtype_w"] = str(fb["q_dtype_w"])
                    self._flydsl_fallbacks.append(fb)
                    print(
                        f"  Fallback (non-flydsl): "
                        f"{fb['kernelName1']}, {fb['kernelName2']}, "
                        f"{fb['us']} us"
                    )
        if len(prorfiles) > 0:
            profile_result = pd.concat(prorfiles)
            profile_result["err"] = profile_result["err"].apply(lambda x: f"{x:.1%}")
            profile_file = f"{AITER_ROOT_DIR}/aiter/configs/profile_fmoe.csv"
            old_profile = self.get_tuned_gemm_list(
                profile_file, profile_result.columns.tolist()
            )
            profile_result = pd.concat([old_profile, profile_result])
            profile_result.to_csv(profile_file, index=False)
        if len(bests) > 0:
            return pd.concat(bests, axis=1).T
        else:
            return pd.DataFrame()

    def pre_process(self, args):
        if args.all:
            self.get_retune_gemm_list(args)
        else:
            self.untunedf = self.get_untuned_gemm_list(args.untune_file)

            if not args.all or args.last:
                self.tunedf = self.get_tuned_gemm_list(
                    self.get_out_file(args.tune_file)
                )
            else:
                self.tunedf = None
            self.untunedf["cu_num"] = self.get_cu_num()
            if args.last:
                self.untunedf = self.untunedf.iloc[-1:]

            elif self.tunedf is not None:
                untunedf_cols = self.untunedf.columns
                mask = self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[untunedf_cols].apply(tuple, axis=1)
                )
                self.untunedf = self.untunedf[~mask]


if __name__ == "__main__":
    key = [
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    resultList = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "tflops",
        "bw",
    ]
    tuner = FmoeTuner("fmoeTuner", key, resultList, "fmoe tuner")
    args = tuner.parse_args()

    tuner.run(args, False)
