# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
from pathlib import Path
import shutil
import re
from moe_cktile2stages_common import (
    kernelInstance,
    get_gemm1_kernels_list,
    get_gemm2_kernels_list,
    get_heuristic_dispatch_template,
)
import sys
from chip_info import get_gfx

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter/jit/utils")  # pip install mode
else:
    AITER_CORE_DIR = os.path.abspath(
        f"{this_dir}/../../aiter/jit/utils"
    )  # develop mode
sys.path.insert(0, AITER_CORE_DIR)


class cktile_moe_2stage_gemm_codegen:
    def __init__(
        self,
        working_path,
        a_dtypes,
        acc_dtype,
        c_dtype,
        quant_type,
        activation,
        mul_routed_weight_stage,
        istune=False,
    ):
        self.init = True
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.dispatchers_path = os.path.join(working_path, "dispatchers")
        self.istune = istune
        self.kernel_name_list = []
        self.a_dtypes = a_dtypes
        self.b_dtypes = ["fp4"]
        self.acc_dtype = acc_dtype.lower()
        self.c_dtype = c_dtype.lower()
        self.quant_type = quant_type
        self.activation = activation
        self.mul_routed_weight_stage = mul_routed_weight_stage

    def get_suffix(self, stage: int) -> str:
        return ("_").join(
            element
            for element in [
                self.quant_type,
                "MulRoutedWeight" if self.mul_routed_weight_stage == stage else "",
                "" if (stage == 2) else self.activation,
            ]
            if element != ""
        )

    def gen_instance(self, k: kernelInstance, a_type):
        INSTANCE_IMPL = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages_common.cuh"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
{k.name}(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.
    int NumTokens = XQ.size(0);
    int M = sorted_ids.size(0);
    int N = WQ.size(1);
    int K = XQ.size(-1);
    int E = WQ.size(0);
    int KBatch = 1;
    int stride_A = K;
    int stride_B = K;
    int stride_C = N / {3 - k.stage}; //gemm1 gate+up need / 2.
    void *sorted_weights_ptr = topk_weight.has_value() ? topk_weight.value().data_ptr() : nullptr;

    {{INSTANCE_CONTENT}}
    return Y;
}}}}

"""
        # default no quant
        scaleGranA = "-1"
        scaleGranB = "-1"
        biasGran = "-1"
        xptr = "nullptr"
        wptr = "nullptr"
        biasptr = "nullptr"
        if k.QuantType == "per_tenser":
            scaleGranA = "0"
            scaleGranB = "0"
            xptr = "static_cast<float>(x_scale.value().data_ptr()[0])"
            wptr = "static_cast<float>(w_scale.value().data_ptr()[0])"
        elif k.QuantType == "per_token":
            scaleGranA = "1"
            scaleGranB = "1"
            xptr = "static_cast<float*>(x_scale.value().data_ptr())"
            wptr = "static_cast<float*>(w_scale.value().data_ptr())"
        elif k.QuantType == "1x32":
            # scaleGranA = "-1"
            scaleGranA = "1, 32"
            scaleGranB = "1, 32"
            biasGran = "1"
            xptr = "x_scale.has_value() ? static_cast<float*>(x_scale.value().data_ptr()) : nullptr"
            wptr = "static_cast<float*>(w_scale.value().data_ptr())"
            biasptr = "static_cast<float*>(exp_bias.value().data_ptr())"

        INSTANCE_CONTENT = f"""auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<{scaleGranA}>{{{xptr}}};
    auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<{scaleGranB}>{{{wptr}}};
    auto exp_bias_dev_ptr = ck_tile::FlatmmScalePointer<{biasGran}>{{{biasptr}}};
    ck_tile::MoeFlatmmHostArgs<decltype(per_a_scale_dev_ptr),
                               decltype(per_b_scale_dev_ptr),
                               decltype(exp_bias_dev_ptr)> kernel_args{{
                reinterpret_cast<const ck_tile::index_t*>(sorted_ids.data_ptr()),
                sorted_weights_ptr,
                reinterpret_cast<const ck_tile::index_t*>(sorted_expert_ids.data_ptr()),
                reinterpret_cast<const ck_tile::index_t*>(max_token_ids.data_ptr()),
                reinterpret_cast<const void*>(XQ.data_ptr()),
                reinterpret_cast<const void*>(WQ.data_ptr()),
                reinterpret_cast<void*>(Y.data_ptr()),
                NumTokens,
                E,
                topk,
                1, // k_batch
                M,
                N,
                K,
                stride_A,
                stride_B,
                stride_C,
                n_padded_zeros.value(),
                k_padded_zeros.value(),
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr,
                exp_bias_dev_ptr
    }};
    using TileConfig = MoeFlatmmConfig<ADataType,
        {k.MPerBlock},
        {k.NPerBlock},
        {k.KPerBlock},
        {k.WAVE_MAP_M},
        {k.WAVE_MAP_N},
        {k.WAVE_TILE_M},
        {k.WAVE_TILE_N},
        {k.WAVE_TILE_K},
        {k.Block_Per_CU}>;
    // Run kernel instance.
    auto stream_config = ck_stream_config{{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()}};
    moe_gemm<TileConfig,
                ADataType,
                BDataType,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                row_major,
                col_major,
                ck_tile::tuple<>,
                row_major,
                {"ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up" if k.stage == 1 else "ck_tile::MoeFlatmmKind::kFFN_gemm2"},
                ck_tile::element_wise::PassThrough
                >(kernel_args, stream_config);
"""

        INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT=(INSTANCE_CONTENT))

        Path(os.path.join(self.impl_path, f"{k.name}.cuh")).write_text(
            INSTANCE_IMPL_str
        )

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "../impl/{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias);

"""

        # if self.istune:
        #     INSTANCE_abI8_dBF16_eBF16 = INSTANCE_template.format(
        #         name=k.name, dtypes="I8, B16"
        #     )
        #     Path(
        #         os.path.join(self.instances_path, f"{k.name}_abI8_dB16_eB16.cpp")
        #     ).write_text(INSTANCE_abI8_dBF16_eBF16)
        # else:
        def fill_template(name, a_type, b_type, acc_type, c_type):
            nonlocal self
            intsance = INSTANCE_template.format(
                name=name, dtypes=f"{a_type}, {b_type}, {acc_type}, {c_type}"
            )
            Path(
                os.path.join(
                    self.instances_path,
                    f"{name}_a{a_type}_b{b_type}_acc{acc_type}_C{c_type}.cpp",
                )
            ).write_text(intsance)

        if (k.QuantType == "1x32") and (a_type in ["bf16", "fp16", "fp8"]):
            fill_template(k.name, a_type, "pk_fp4", self.acc_dtype, self.c_dtype)
        else:
            for CDtype in ["bf16", "fp16"]:
                for ABDtype in ["fp8"]:  # "bf16", "fp16",
                    for AccDtype in ["float"]:
                        fill_template(k.name, ABDtype, ABDtype, AccDtype, CDtype)
                        # intsance = INSTANCE_template.format(
                        #     name=k.name, dtypes=f"{ABDtype},  {AccDtype}, {CDtype}"
                        # )
                        # Path(
                        #     os.path.join(
                        #         self.instances_path,
                        #         f"{k.name}_ab{ABDtype}_acc{AccDtype}_C{CDtype}.cpp",
                        #     )
                        # ).write_text(intsance)

    """genarete heuristic dispatch"""

    def gen_heuristic_dispatch(self, tag, kernels_dict):
        HEURISTIC_template = get_heuristic_dispatch_template(tag)
        # print(HEURISTIC_template)

        def validate_and_format(template: str, mapping: dict) -> str:
            # check all format element in dict.
            str_mapping = {str(key): value.name for key, value in mapping.items()}
            cleaned_template = template.replace("{{", "").replace("}}", "")
            placeholders = re.findall(r"\{([^{}]*)\}", cleaned_template)
            missing = [p for p in placeholders if p not in str_mapping]
            # print(placeholders)
            # print(str_mapping)
            if missing:
                for mis in missing:
                    placeholders.remove(mis)
            # result = template
            # for placeholder in placeholders:
            #     result = result.replace(placeholder, str_mapping[placeholder])
            # return result
            return template.format(**{k: v for k, v in str_mapping.items()})

        # create heuristic heirarchy
        with open(
            os.path.join(
                self.dispatchers_path, f"moe_cktile2stages_heuristic_dispatch_{tag}.h"
            ),
            "w",
        ) as f:
            f.write(validate_and_format(HEURISTIC_template, kernels_dict))
            # arch = get_gfx()
            # inst_k = "32" if self.quant_type == "1x32" else ("128" if arch == "gfx950" else "64")
            # f.write(
            #     HEURISTIC_template.format(
            #         inst_k=inst_k,
            #         suffix1 = self.get_suffix(1),
            #         suffix2 = self.get_suffix(2)
            #     )
            # )

    """genarete heuristic dispatch header for multi dtype"""

    def gen_heuristic_dispatch_header(self, tags):
        HEURISTIC_dispatch_header = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"

"""
        for tag in tags:
            HEURISTIC_headers = f"""#include "./dispatchers/moe_cktile2stages_heuristic_dispatch_{tag}.h"
"""
            HEURISTIC_dispatch_header += HEURISTIC_headers

        HEURISTIC_function = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_cktile2stages.h"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
MoeKernel moe_gemm1_heuristic_dispatch(int M, int N, int K, int block_m);

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
MoeKernel moe_gemm2_heuristic_dispatch(int M, int N, int K, int block_m);
"""
        # create heuristic heirarchy
        with open(
            os.path.join(self.working_path, "moe_cktile2stages_heuristic_dispatch.h"),
            "w",
        ) as f:
            f.write(HEURISTIC_dispatch_header)
        with open(
            os.path.join(
                self.dispatchers_path, "moe_cktile2stages_heuristic_dispatch_common.h"
            ),
            "w",
        ) as f:
            f.write(HEURISTIC_function)

    """generate lookup.h linking MNK/datatype to specific instance"""

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// #ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(ABTYPE, ACCTYPE, CTYPE)                                                                           \\
   {                                                                                                                            \\"""

        LOOKUP_template = """
            {{{MNK},                                                      \\
            {kernel_name}<ABTYPE, ACCTYPE, CTYPE>}},                      \\"""

        LOOKUP_end = """
   }

// #endif // USE_ROCM
"""
        with open(
            os.path.join(self.working_path, "moe_cktile2stages_lookup.h"), "w"
        ) as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                print(":", k.name)
                # if not tunning, tuned mnk = {stage, m, n, k}
                if not self.istune and (
                    isinstance(mnk, tuple) and (len(mnk) == 4) and mnk[1] > 0
                ):
                    f.write(
                        LOOKUP_template.format(
                            MNK="{"
                            + (", ").join(map(lambda x: str(x), list(mnk)))
                            + "}",
                            kernel_name=k.name,
                        )
                    )
                # if tunning, mnk = -1,-2.....
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(MNK=mnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    """generate manifest.h for instance header"""

    def gen_manifest_head(self):
        MAINFEST_head = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// #ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>
"""
        MAINFEST_template = """
// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
{kernel_name}(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias);
"""
        MAINFEST_end = """

// endif // USE_ROCM
"""

        with open(
            os.path.join(self.working_path, "moe_cktile2stages_manifest.h"), "w"
        ) as f:
            f.write(MAINFEST_head)
            for k_name in self.kernel_name_list:
                f.write(MAINFEST_template.format(kernel_name=k_name))
            f.write(MAINFEST_end)

    """generate all instances and headers"""

    def gen_instances(self, tag, kernels_dict, a_type):
        if self.init:
            if os.path.exists(self.impl_path):
                shutil.rmtree(self.impl_path)
            os.mkdir(self.impl_path)
            if os.path.exists(self.instances_path):
                shutil.rmtree(self.instances_path)
            os.mkdir(self.instances_path)
            if os.path.exists(self.dispatchers_path):
                shutil.rmtree(self.dispatchers_path)
            os.mkdir(self.dispatchers_path)

        self.init = False

        for mnk, k in kernels_dict.items():
            self.gen_instance(k, a_type)
            if k.name not in self.kernel_name_list:
                self.kernel_name_list.append(k.name)

        self.gen_lookup_dict(kernels_dict)
        self.gen_heuristic_dispatch(tag, kernels_dict)


# def get_tune_dict(tune_dict_csv):
#     tune_dict = default_kernels_dict
#     if os.path.exists(tune_dict_csv):
#         tune_df = pd.read_csv(tune_dict_csv)
#         if torch.cuda.is_available():
#             gpu = torch.cuda.current_device()
#             device_properties = torch.cuda.get_device_properties(gpu)
#             cu_num = device_properties.multi_processor_count
#             tune_df = tune_df[tune_df["cu_num"] == cu_num].reset_index()
#         for i in range(len(tune_df)):
#             M = tune_df.loc[i, "M"]
#             N = tune_df.loc[i, "N"]
#             K = tune_df.loc[i, "K"]
#             kid = tune_df.loc[i, "kernelId"]
#             tune_dict[(M, N, K)] = kernels_list[kid]
#     return tune_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ck_tile 2stage gemm instance."
    )

    # Add arguments
    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    parser.add_argument(
        "-a",
        "--a_dtype",
        nargs="*",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16"],
        help="select input dtype",
    )

    parser.add_argument(
        "-b",
        "--b_dtype",
        nargs="*",
        required=False,
        type=str,
        choices=["f8", "i8", "f16", "b16", "i4"],
        help="select weight dtype",
    )

    parser.add_argument(
        "-c",
        "--c_dtype",
        default="b16",
        required=False,
        type=str,
        choices=["f16", "b16"],
        help="select out dtype",
    )

    parser.add_argument(
        "-q",
        "--quant_type",
        default="per_tensor",
        required=False,
        type=str,
        choices=[
            "per_tensor",
            "per_token",
            "1x32",
            "128x128",
            "no",
        ],
        help="select quant_type",
    )

    parser.add_argument(
        "-act",
        "--activation",
        default="silu",
        required=False,
        type=str,
        choices=["silu", "gelu"],
        help="select activation",
    )

    parser.add_argument(
        "-m",
        "--mul_routed_weight_stage",
        default=2,
        required=False,
        type=int,
        choices=[1, 2],
        help="select quant_type",
    )

    args = parser.parse_args()

    # # build all
    # if args.b_dtype is None:
    #     # quanted moe
    #     b_quant_dtypes = ["f8"]
    #     c_dtypes = ["f16", "b16"]
    #     acts = ["silu"] #, "gelu"]
    #     general_quant_l = ["per_tensor", "per_token"]
    #     for b_dtype, c_dtype, act, quant in itertools.product(
    #         b_quant_dtypes, c_dtypes, acts, general_quant_l
    #     ):
    #         a_dtype = b_dtype
    #         codegen = cktile_moe_2stage_gemm_codegen(
    #             args.working_path,
    #             a_dtype,
    #             b_dtype,
    #             c_dtype,
    #             quant,
    #             act,
    #         )
    #         codegen.generate_instance_and_lookUpTable()

    #     # no-quant moe
    #     b_quant_dtypes = [
    #         "f16",
    #         "b16",
    #     ]
    #     for (
    #         b_dtype,
    #         act,
    #     ) in itertools.product(b_quant_dtypes, acts):
    #         c_dtype = a_dtype = b_dtype

    #         codegen = cktile_moe_2stage_gemm_codegen(
    #             args.working_path,
    #             a_dtype,
    #             b_dtype,
    #             c_dtype,
    #             "no",
    #             act,
    #         )
    #         codegen.generate_instance_and_lookUpTable()
    # else:

    # single UT
    # a_type = "fp8"
    # b_type = "fp8"
    # quant_type = "per_token"

    a_types = ["bf16"]
    if get_gfx() == "gfx950":
        a_types.append("fp8")
    b_type = "fp4"
    quant_type = "1x32"

    acc_type = "float"
    c_type = "bf16"
    act_type = "silu"
    codegen = cktile_moe_2stage_gemm_codegen(
        args.working_path, a_types, acc_type, c_type, quant_type, act_type, 2, False
    )
    # gen all instances for gemm1 and gemm2
    tags = []
    kernel_list = []
    for a_type in a_types:
        _, gemm1_kernel_list = get_gemm1_kernels_list(
            a_type,
            b_type,
            quant_type,
            act_type,
            False,
        )
        tag, gemm2_kernel_list = get_gemm2_kernels_list(
            a_type,
            b_type,
            quant_type,
            "",
            True,
        )
        # merge gemm1/gemm2 dict with key = {stage, key}
        kernel_dict_merge = {
            **{(1, key): value for key, value in gemm1_kernel_list.items()},
            **{(2, key): value for key, value in gemm2_kernel_list.items()},
        }
        # print(kernel_dict_merge)
        codegen.gen_instances(tag, kernel_dict_merge, a_type)
        tags.append(tag)
    codegen.gen_heuristic_dispatch_header(tags)
    codegen.gen_manifest_head()
