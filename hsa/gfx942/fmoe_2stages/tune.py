# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
import pandas as pd
import os
import sys
from aiter import QuantType
from aiter.jit.core import get_asm_dir, AITER_CSRC_DIR, AITER_CONFIG_FMOE
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    asm_stage1,
    torch_moe_stage1,
    torch_moe_stage2,
    torch_moe,
)
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, dtype2str_dict
from aiter.ops.shuffle import shuffle_weight
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


sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list


torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


class FmoeTuner(TunerCommon):

    ARG_DEFAULTS = {
        "verbose": False,
        "tune_file": f"{AITER_CONFIG_FMOE}",
        "untune_file": "aiter/configs/untuned_fmoe.csv",
        "errRatio": 0.5,
        "batch": 100,
        "profile_file": "",  # for all results
    }

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
    ):
        inter_dim = w1_qt_shffle_ck.shape[1] // 2
        token_num = a1_qt.shape[0]
        out = torch.empty(
            (token_num, topk, inter_dim),
            dtype=dtype,
            device=a1_qt.device,
        )
        out = ck_moe_stage1_fwd(
            a1_qt,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            kernelName,
            w1_scale,
            a1_scale,
            blockM,
            sorted_weights,
            q_type,
            act_type,
        )
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
            and (q_dtype_a in [dtypes.bf16, dtypes.fp16])
            and q_dtype_w == dtypes.fp4x2
        ):  # a16w4
            a1_qt = input.to(dtype)
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
        if q_dtype_w == torch.int4:
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
        else:
            w1_qt_shffle_ck = w1_qt_shffle
            w2_qt_shffle_ck = w2_qt_shffle
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
        if stage == 1:
            if not doweight_stage1:
                sorted_weights = None
            if q_type == QuantType.per_1x32:
                a1_scale_fp4_sort = moe_mxfp4_sort(
                    a1_scale,  # a1_scale[: token * topk, :].view(token, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=blockM,
                )
            else:
                a1_scale_fp4_sort = a1_scale

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
                w1_scale_aiter,
            )
        elif stage == 2:
            ref1 = FmoeTuner.run_torch_moe_stage1(
                a1_qt,
                w1_qt,
                w2_qt,
                topk_weights,
                topk_ids,
                a1_scale=a1_scale,
                w1_scale=w1_scale,
                dtype=dtype,
                activation=act_type,
                quant_type=q_type,
                doweight_stage1=doweight_stage1,
                topk=topk,
            )
            if q_type == QuantType.per_1x128:
                ref1, ref_scale = aiter.pertoken_quant(
                    ref1.view(ref1.shape[0], -1, 128), quant_dtype=q_dtype_a
                )
                ref1 = ref1.view(ref1.shape[0], topk, -1)
                ref_scale = ref_scale.view(token, -1)
                a2_qt = ref1
                a2_scale = ref_scale
            else:
                torch_quant = aiter.get_torch_quant(q_type)
                a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
            a2_qt = a2_qt.view(token, topk, -1)
            if doweight_stage1:
                sorted_weights = None
            if q_type == QuantType.per_1x32:
                a2_scale_mxfp4_sort = moe_mxfp4_sort(
                    a2_scale[: token * topk, :].view(token, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=blockM,
                )
            else:
                a2_scale_mxfp4_sort = a2_scale
            return (
                a2_qt,  # 0
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
                w2_scale_aiter,
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
        dtype,
        activation,
        quant_type,
        doweight_stage1,
        topk,
    ):
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
            doweight=doweight_stage1,
        )
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
        dtype,
        quant_type,
        doweight_stage1,
    ):
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
            # gose to quant D_w8a8/w8a8
            # blk_n, blk_k = scale_blks
            # expert, nblk_n, nblk_k = fc1_scale.shape
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
        if us == self.INVALID_TIME:
            return -1, -1
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
        chip_name = get_gfx()
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
        # blockMs = [32, 64, 128]
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
            if use_g1u1 and q_dtype_w != torch.int4:
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
        # blockMs = [32, 64, 128]
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

        _, ck_stage1_kernels = get_gemm1_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,
            int(q_type),
            str(act_type).split(".")[-1].lower(),
            doweight_stage1,
            False,  # bpreshuffle
        )
        _, ck_stage2_kernels = get_gemm2_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,
            int(q_type),
            not doweight_stage1,
            False,  # bpreshuffle
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
                                # [a1_qt, w1_qt, w2_qt, topk_weights, topk_ids, a1_scale, w1_scale]
                                [0, 10, 11, 12, 13, 3, 4],
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

                for kernel in ck_stage2_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
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
                            (
                                [0, 10, 11, 12, 13, 3, 4],
                                dtype,
                                q_type,
                                doweight_stage1,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            True,
                        )
                    )
        return tasks_ck

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        mp_num = args.mp
        blockMs = [16, 32, 64, 128]
        args = self.keys
        print(untunedf[args])
        tasks = []
        tasks_ck = []
        task_1stage = []
        in_data = []
        for line in untunedf[args].values:
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
            # info = line
            dtype = eval(dtype)
            q_dtype_a = eval(q_dtype_a)
            q_dtype_w = eval(q_dtype_w)
            q_type = eval(q_type)
            q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
            print("\nStart tuning", line)
            if get_gfx() not in ["gfx950"] and q_type == aiter.QuantType.per_1x32:
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
            task_1stage.extend(self.gen_1stage_asm_task(info))
            if tasks is None and tasks_ck is None and task_1stage is None:
                print("no moe solution can tune for ", line)
                continue
            print(
                f"stage1 asm tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}, task_1stage is {len(task_1stage)}"
            )
        in_data.append((len(tasks) + len(tasks_ck) + len(task_1stage), ()))
        rets = []
        if len(tasks) + len(tasks_ck) + len(task_1stage) > 0:
            ### shape_grouped should be False as multiple stages
            rets = mp_tuner(
                tasks + tasks_ck + task_1stage, in_data, mp_num, True, False
            )
        if not rets:
            print("no shape to tune or no solution found")
            return []
        else:
            return rets

    def result_to_csv(self, results, file, concat=False):
        old_tunedf = self.get_tuned_gemm_list(file)
        resultdf = self.update_tunedf(old_tunedf, results)
        self.success = pd.concat([self.success, results], ignore_index=True)
        resultdf["run_1stage"] = resultdf["run_1stage"].astype(int)
        if results is not None:
            resultdf = resultdf.astype(str).drop_duplicates(
                subset=self.keys,
                keep="last",
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
            profileDF = []
            for (stage, kernelName, block_m), us, err in rets:
                # if us == float("inf"):
                #    continue
                # if err > args.errRatio:
                #    continue
                tflops, bw = self.calculate((key, stage, kernelName, block_m, us, err))
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
                        q_dtype_w if q_dtype_w != torch.int4 else "torch.int4",
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        block_m,
                        0,
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
                (profileDF["err"] < args.errRatio) & (profileDF["us"] != float("-inf"))
            ]
            profileDF = profileDF.sort_values("us").drop_duplicates(
                ["stage", "block_m"], keep="first"
            )
            stage1_profileDF = profileDF[profileDF["stage"] == "stage1"].drop(
                columns=["stage"], axis=1
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
                columns=["stage", "ksplit"], axis=1
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
                columns=["stage"], axis=1
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
                        q_dtype_w if q_dtype_w != torch.int4 else "torch.int4",
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
                f"Tuning result for {key} is {best_one['block_m'] ,best_one['kernelName1'], best_one['kernelName2'], best_one['err1'], best_one['err2'],  best_one['run_1stage']} {best_one['us']} us, {best_one['tflops']} TFLOPS, {best_one['bw']} GB/s"
            )
            best_one["act_type"] = str(best_one["act_type"])
            best_one["q_type"] = str(best_one["q_type"])
            best_one["dtype"] = str(best_one["dtype"])
            best_one["q_dtype_a"] = str(best_one["q_dtype_a"])
            best_one["q_dtype_w"] = str(best_one["q_dtype_w"])
            bests.append(best_one)
        if len(prorfiles) > 0:
            profile_result = pd.concat(prorfiles)
            profile_result["err"] = profile_result["err"].apply(lambda x: f"{x:.1%}")
            profile_file = f"aiter/configs/profile_fmoe.csv"
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
