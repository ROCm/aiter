# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import itertools
from pathlib import Path
import shutil
import re
from moe_cktile2stages_common import (
    act_dict,
    dtype_dict,
    kernelInstance,
    get_gemm1_kernels_list,
    get_gemm2_kernels_list,
    get_heuristic_dispatch_template,
)
import sys
from chip_info import get_gfx
from aiter.utility.amd_buffer_coherence import (
    BufferCoherenceType,
    BufferCoherenceMapper,
)

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
        a_dtype,
        acc_dtype,
        c_dtype,
        quant_type,
        activation,
        mul_routed_weight_stage,
        is_split_k,
        b_nt_type,
        istune=False,
    ):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.dispatchers_path = os.path.join(working_path, "dispatchers")
        self.manifests_path = os.path.join(working_path, "manifests")
        self.istune = istune
        self.kernel_name_list = []
        self.a_dtype = a_dtype
        self.b_dtype = "fp4"
        self.acc_dtype = acc_dtype.lower()
        self.c_dtype = c_dtype.lower()
        self.quant_type = quant_type
        self.is_split_k = is_split_k
        self.activation = act_dict[activation]
        self.b_nt_type = b_nt_type
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

    def gen_instance(self, k: kernelInstance):
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
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch)
{{{{
    // The smallest kernel we have available. Works well for memory bound shapes.
    int NumTokens = XQ.size(0);
    int M = sorted_ids.size(0);
    int N = WQ.size(1);
    int K = XQ.size(-1);
    int E = WQ.size(0);
    int KBatch = k_batch.has_value() ? k_batch.value() : 1;
    int stride_A = K;
    int stride_B = K;
    int stride_C = KBatch > 1 ? N : N / {3 - k.stage}; //gemm1 gate+up need / 2.
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
            scaleGranA = "1, 32, ck_tile::e8m0_t"
            scaleGranB = "1, 32, ck_tile::e8m0_t"
            biasGran = "1"
            xptr = "x_scale.has_value() ? static_cast<ck_tile::e8m0_t*>(x_scale.value().data_ptr()) : nullptr"
            wptr = "static_cast<ck_tile::e8m0_t*>(w_scale.value().data_ptr())"
            biasptr = "static_cast<float*>(exp_bias.has_value() ? exp_bias.value().data_ptr() : nullptr)"

        if act_dict[k.ActOP] != 2:
            biasGran = "-1"

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
                KBatch, // k_batch
                M,
                N,
                K,
                stride_A,
                stride_B,
                stride_C,
                n_padded_zeros.has_value() ? n_padded_zeros.value() : 0,
                k_padded_zeros.has_value() ? k_padded_zeros.value() : 0,
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
                {"ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k" if self.is_split_k else
                "ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up" if k.stage == 1 else
                "ck_tile::MoeFlatmmKind::kFFN_gemm2"},
                ck_tile::element_wise::PassThrough,
                {act_dict[k.ActOP]},
                {k.b_nt_type}
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
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

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
            fill_template(k.name, self.a_dtype, "pk_fp4", self.acc_dtype, self.c_dtype)
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

    def gen_heuristic_dispatch(self, tag, dict):
        HEURISTIC_template = get_heuristic_dispatch_template(tag)
        # print(HEURISTIC_template)

        def validate_and_format(template: str, mapping: dict) -> str:
            # check all format element in dict.
            str_mapping = {
                "(a_data_type)": dtype_dict[self.a_dtype],
                "(b_data_type)": dtype_dict[self.b_dtype],
                "(acc_data_type)": dtype_dict[self.acc_dtype],
                "(c_data_type)": dtype_dict[self.c_dtype],
                "(activation)": self.activation,
                "(has_bias)": "true" if self.activation == 2 else "false",
                "(split_k)": "true" if self.is_split_k else "false",
                "(b_nt_type)": str(self.b_nt_type),
            }
            format_args = {str(key): value.name for key, value in mapping.items()}
            str_mapping.update(format_args)
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

        _, k = next(iter(dict.items()))
        # create heuristic heirarchy
        with open(
            os.path.join(
                self.dispatchers_path, f"{k.dispatch_suffix}_heuristic_dispatch_{tag}.h"
            ),
            "w",
        ) as f:
            f.write(validate_and_format(HEURISTIC_template, dict))

        return f"./dispatchers/{k.dispatch_suffix}_heuristic_dispatch_{tag}.h"

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

    def gen_manifest_head(self, tag, kernels_dict):
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
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);
"""
        MAINFEST_end = """

// endif // USE_ROCM
"""
        _, k0 = next(iter(kernels_dict.items()))
        with open(
            os.path.join(self.manifests_path, f"{k0.dispatch_suffix}_manifest_{tag}.h"),
            "w",
        ) as f:
            f.write(MAINFEST_head)
            for k_name in self.kernel_name_list:
                f.write(MAINFEST_template.format(kernel_name=k_name))
            f.write(MAINFEST_end)

        return f"./manifests/{k0.dispatch_suffix}_manifest_{tag}.h"

    """generate all instances and headers"""

    def gen_instances(self, tag, kernels_dict):
        for mnk, k in kernels_dict.items():
            self.gen_instance(k)
            if k.name not in self.kernel_name_list:
                self.kernel_name_list.append(k.name)

        self.gen_lookup_dict(kernels_dict)
        self.gen_heuristic_dispatch(tag, kernels_dict)
        return self.gen_heuristic_dispatch(tag, kernels_dict), self.gen_manifest_head(
            tag, kernels_dict
        )


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


def generate_common_header(working_path, dispatch_files, manifest_files):

    common_header = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

"""
    dispatch_header = common_header
    manifest_header = common_header

    for file_name in sorted(dispatch_files):
        include_path = f'"{file_name}"'
        dispatch_header += f"#include {include_path}\n"

    dispatch_common_header_path = os.path.join(
        working_path, "moe_cktile2stages_heuristic_dispatch_common.h"
    )
    with open(dispatch_common_header_path, "w") as f:
        f.write(dispatch_header)

    for file_name in sorted(manifest_files):
        include_path = f'"{file_name}"'
        manifest_header += f"#include {include_path}\n"

    manifest_common_header_path = os.path.join(
        working_path, "moe_cktile2stages_manifest_common.h"
    )
    with open(manifest_common_header_path, "w") as f:
        f.write(manifest_header)

    """genarete heuristic dispatch header for multi dtype"""


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
        choices=["silu", "gelu", "swiglu"],
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
    act_types = ["silu", "swiglu"]
    c_dtypes = ["bf16"]
    is_split_k_l = [True, False]

    b_nt_type_l = [
        BufferCoherenceMapper.map_to_arch_value(BufferCoherenceType.DEFAULT),
        BufferCoherenceMapper.map_to_arch_value(BufferCoherenceType.WAVE_NT),
    ]

    impl_path = os.path.join(args.working_path, "impl")
    instances_path = os.path.join(args.working_path, "instances")
    dispatchers_path = os.path.join(args.working_path, "dispatchers")
    manifests_path = os.path.join(args.working_path, "manifests")

    if os.path.exists(impl_path):
        shutil.rmtree(impl_path)
    os.mkdir(impl_path)
    if os.path.exists(instances_path):
        shutil.rmtree(instances_path)
    os.mkdir(instances_path)
    if os.path.exists(dispatchers_path):
        shutil.rmtree(dispatchers_path)
    os.mkdir(dispatchers_path)
    if os.path.exists(manifests_path):
        shutil.rmtree(manifests_path)
    os.mkdir(manifests_path)


    gen_dispatch_files = []
    gen_manifest_files = []
    tags = []
    kernel_list = []

    for a_type, c_dtype, act_type, is_split_k, b_nt_type in itertools.product(
        a_types, c_dtypes, act_types, is_split_k_l, b_nt_type_l
    ):
        has_bias = True if act_type == "swiglu" else False

        # a8w8 do not support
        if a_type in ["fp8", "bf8"] and is_split_k:
            continue
        codegen = cktile_moe_2stage_gemm_codegen(
            args.working_path,
            a_type,
            acc_type,
            c_dtype,
            quant_type,
            act_type,
            2,
            is_split_k,
            b_nt_type,
            False,
        )
        # gen all instances for gemm1 and gemm2
        _, gemm1_kernel_list = get_gemm1_kernels_list(
            a_type,
            b_type,
            quant_type,
            act_type,
            False,
            has_bias,
            is_split_k,
            b_nt_type,
        )
        tag, gemm2_kernel_list = get_gemm2_kernels_list(
            a_type,
            b_type,
            quant_type,
            "no",
            True,
            has_bias,
            b_nt_type,
        )
        # merge gemm1/gemm2 dict with key = {stage, key}
        kernel_dict_merge = {
            **{(1, key): value for key, value in gemm1_kernel_list.items()},
            **{(2, key): value for key, value in gemm2_kernel_list.items()},
        }
        dispatch_file, manifest_file = codegen.gen_instances(tag, kernel_dict_merge)

        gen_dispatch_files.append(dispatch_file)
        gen_manifest_files.append(manifest_file)

    generate_common_header(args.working_path, gen_dispatch_files, gen_manifest_files)
