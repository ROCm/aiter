# SPDX-License-Identifier: MIT
# Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass

def get_if_str(idx, total, last_else=True):
    if idx == 0:
        return "if"
    else:
        return "else if"


DATA_TYPE_MAP = {
    "float32": "float",
    "float64": "double",
    "int32": "int",
    "int64": "long long",
    "bool": "bool",
    "float16": "torch::Half",
    "bfloat16": "torch::BFloat16",
}

TORCH_TYPE_MAP = {
    "float32": "torch::kFloat32",
    "float64": "torch::kFloat64",
    "int32": "torch::kInt32",
    "int64": "torch::kInt64",
    "bool": "torch::kBool",
    "float16": "torch::kHalf",
    "bfloat16": "torch::kBFloat16",
}

OPERATOR_MAP = {
    "add": "aiter::AddOp",
    "sub": "aiter::SubOp",
    "mul": "aiter::MulOp",
    "div": "aiter::DivOp",
}


class BinaryOpCodegen:
    API_COMMON_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <torch/extension.h>
#include "binary_operator.cuh"

template <typename Op, typename T0, typename T1>
void binary_op_impl(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  int dim = input.dim();

  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;   // (m, n, k), (1, n, k)
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;   // (m, n, k), (m, 1, k)
  constexpr uint32_t PATTERN_CONTIGUOUS = 4;
  constexpr uint32_t PATTERN_BROADCAST_2 = 5;   // (m, n, k), (m, n, 1)
  constexpr uint32_t PATTERN_BROADCAST_3 = 6;   // (m, n, k), (   n, 1)
  constexpr uint32_t PATTERN_BROADCAST_4 = 7;   // (m, n, k), (m, 1, 1)

  // contiguous case
  if (!is_support)
  {
    is_support = true;
    is_support &= (input.dim() == other.dim());
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    if (input.dim() == 1)
    {
      is_support &= input.numel() % 128 == 0;
    }
    for (int i = 0; i < input.dim() && is_support; ++i)
    {
      is_support &= (input.size(i) == other.size(i));
    }
    pattern = is_support ? PATTERN_CONTIGUOUS : 0;
  }

  if (!is_support && (dim == 3 || other.dim() == 3))
  {
    // transpose case
    if (input.is_contiguous() != other.is_contiguous())
    {
      auto tensor_not_conti = input.is_contiguous() ? other : input;
      order_flag = !input.is_contiguous() ? true : false;
      is_support = true;
      // avoid broadcast
      is_support &= input.dim() == other.dim();
      is_support &= input.size(0) == other.size(0);
      is_support &= input.size(1) == other.size(1);
      is_support &= input.size(2) == other.size(2);
      is_support &= tensor_not_conti.stride(1) == 1;
      pattern = is_support ? PATTERN_TRANSPOSE : 0;
    }
    // broadcast case
    else if (input.is_contiguous() && other.is_contiguous())
    {
      is_support = false;
      // input tensor dim and other tensor dim both equal to 3
      if (input.dim() == other.dim())
      {
        // broadcast at dim0 or dim1 or dim2
        auto broadcast_3d_case = [&] (int bcast_dim)
        {
          constexpr int bcast_pattern[3] = {PATTERN_BROADCAST_0, PATTERN_BROADCAST_1, PATTERN_BROADCAST_2};
          if (!is_support && (input.size(bcast_dim) == 1 || other.size(bcast_dim)) && input.size(bcast_dim) != other.size(bcast_dim))
          {
            is_support = true;
            for (int i = 0; i < 3; ++i)
            {
              if (bcast_dim != i) is_support &= input.size(i) == other.size(i);
            }
            is_support &= input.size(bcast_dim) == 1 ? other.size(bcast_dim) != 1 : true;
            pattern = is_support ? bcast_pattern[bcast_dim] : 0;
            order_flag = input.size(bcast_dim) != 1 ? true : false;
            if (bcast_dim == 1) order_flag = !order_flag;
          }
        };
        // (m, n, k), (1, n, k) or (1, n, k), (m, n, k)
        broadcast_3d_case(0);
        // (m, n, k), (m, 1, k) or (m, 1, k), (m, n, k)
        broadcast_3d_case(1);
        // (m, n, k), (m, n, 1) or (m, n, 1), (m, n, k)
        broadcast_3d_case(2);

        // broadcast in last 2 dim
        bool first_dim_eq = input.size(0) == other.size(0);
        bool input_bcast_last2dim = input.size(1) == 1 && input.size(2) == 1;
        bool other_bcast_last2dim = other.size(1) == 1 && other.size(2) == 1;
        if (first_dim_eq && (input_bcast_last2dim || other_bcast_last2dim))
        {
          is_support = true;
          order_flag = other_bcast_last2dim ? true : false;
          pattern = PATTERN_BROADCAST_4;
        }
      }
      // (m, n, k), (n, 1) or (n, 1), (m, n, k)
      else if (input.dim() == 2 || other.dim() == 2)
      {
        is_support = true;
        if (input.dim() == 2)
        {
          is_support &= input.size(0) == other.size(1);
          is_support &= input.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = false;
        }
        else
        {
          is_support &= other.size(0) == input.size(1);
          is_support &= other.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = true;
        }
      }
      // (m, n, k), (k) or (k), (m, n, k)
      // (m, n, k), (1) or (1), (m, n, k)
      else if (input.dim() == 1 || other.dim() == 1)
      {
        if (other.dim() == 1)
        {
          if (other.size(0) == 1 || (other.size(0) == input.size(2) && input.size(2) % (128 / input.element_size()) == 0))
          {
            if (input.numel() % (256 * 8 * 16 / input.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = true;
            }
          }
        }
        else
        {
          if (input.size(0) == 1 || (input.size(0) == other.size(2) && other.size(2) % (128 / other.element_size()) == 0))
          {
            if (other.numel() % (256 * 8 * 16 / other.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = false;
            }
          }
        }
      }
    }
  }

  if (!is_support && input.dim() != 3 && other.dim() != 3)
  {
    if (input.dim() == other.dim())
    {
      std::vector<int> bcast_dim_index = {};
      for (int i = 0; i < input.dim(); ++i)
      {
        // broadcast condition
        if (input.size(i) != other.size(i) && (input.size(i) == 1 || other.size(i) == 1))
        {
          bcast_dim_index.push_back(i);
        }
      }
      if (bcast_dim_index.size() == 1 && bcast_dim_index[0] != 0 && bcast_dim_index[0] != input.dim() - 1)
      {
        is_support = true;
        pattern = PATTERN_BROADCAST_1;
        order_flag = other.size(bcast_dim_index[0]) == 1 ? true : false;
      }
    }
  }

  // hip does not support double
  if (input.dtype() == torch::kDouble || other.dtype() == torch::kDouble)
  {
    is_support = false;
  }

  if (is_support)
  {
    switch (pattern)
    {
      case PATTERN_TRANSPOSE:
        binary_operation_process<1, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_0:
        binary_operation_process<2, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_1:
        binary_operation_process<3, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_CONTIGUOUS:
        binary_operation_process<4, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_2:
        binary_operation_process<5, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_3:
        binary_operation_process<6, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_4:
        binary_operation_process<7, Op, T0, T1>(input, other, output, order_flag);
        break;
      default:
        return ;
    }
    return ;
  }
  else
  {
  output = aiter::aten_compute<Op>(input, other);
  return ;
  }
}
"""

    API_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "binary_op_api_common.hpp"
void binary_op_dispatch(const std::string& op_type, 
                       torch::Tensor &input, 
                       torch::Tensor &other, 
                       torch::Tensor &output) {{
    // Dispatch based on operator and input types
{F_dispatch}
    //AT_ERROR("Unsupported operator or dtype combination");
}}
"""

    API_PER_OPERATOR = """    {F_if}(op_type == "{F_op_type}") {{
{F_per_dtype}
    }}
"""

    API_PER_DTYPE = """        {F_if}(input.scalar_type() == {F_in0_type} && other.scalar_type() == {F_in1_type}) {{
            binary_op_impl<{F_op_cpp}, {F_in0_cpp}, {F_in1_cpp}>(input, other, output);
        }}
"""

    INSTANCE_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "binary_op_api_common.hpp"

// Explicit instantiation
{F_instance_def}
"""

    def __init__(self, working_path, dtypes, optype):
        self.working_path = working_path
        self.input_dtype = dtypes.split("_")[0]
        self.other_dtype = dtypes.split("_")[1]
        self.op_type = optype
        print("@@@@@", self.input_dtype, self.other_dtype, self.op_type)

    @dataclass
    class h_traits:
        F_op_type: str
        F_in0_type: str
        F_in1_type: str

        @property
        def trait_name(self) -> str:
            return f"{OPERATOR_MAP[self.F_op_type]}, {DATA_TYPE_MAP[self.F_in0_type]}, {DATA_TYPE_MAP[self.F_in1_type]}"
        @property
        def call_name(self) -> str:
          return f'binary_op_impl<{self.trait_name}>'

        @property
        def def_name(self) -> str:
            return f"template void binary_op_impl<{self.trait_name}>(torch::Tensor&, torch::Tensor&, torch::Tensor&);"

    @dataclass
    class h_instance:
        F_op_type: str
        F_dtype_pair: str  # "in0_type,in1_type"
        instance_list: List[Any]

        @property
        def name(self) -> str:
            in0, in1 = self.F_dtype_pair.split(",")
            return f"binary_op_{self.F_op_type}_{in0}_{in1}"

        @property
        def content(self) -> str:
            instance_defs = "\n".join(ins.def_name for ins in self.instance_list)
            return BinaryOpCodegen.INSTANCE_BASE.format(F_instance_def=instance_defs)

    def content_api(self) -> str:
        op_dict = {}
        blobs = self.get_blobs()

        for blob in blobs:
            if blob.F_op_type not in op_dict:
                op_dict[blob.F_op_type] = {}
            if blob.F_dtype_pair not in op_dict[blob.F_op_type]:
                op_dict[blob.F_op_type][blob.F_dtype_pair] = blob

        dispatch_str = ""
        for i_op, (op_type, dtype_blobs) in enumerate(op_dict.items()):
            dtype_str = ""
            for i_dtype, (dtype_key, blob) in enumerate(dtype_blobs.items()):
                in0, in1 = dtype_key.split(",")
                dtype_str += self.API_PER_DTYPE.format(
                    F_if=get_if_str(i_dtype, len(dtype_blobs)),
                    F_in0_type=TORCH_TYPE_MAP[in0],
                    F_in1_type=TORCH_TYPE_MAP[in1],
                    F_op_cpp=OPERATOR_MAP[op_type],
                    F_in0_cpp=DATA_TYPE_MAP[in0],
                    F_in1_cpp=DATA_TYPE_MAP[in1],
                )

            dispatch_str += self.API_PER_OPERATOR.format(
                F_if=get_if_str(i_op, len(op_dict)),
                F_op_type=op_type,
                F_per_dtype=dtype_str,
            )

        return self.API_BASE.format(F_traits_define="", F_dispatch=dispatch_str)

    def get_blobs(self):
        operators = [self.op_type]
        dtype_combinations = [(str(self.input_dtype), str(self.other_dtype))]

        blobs = []
        for op in operators:
            for in0, in1 in dtype_combinations:
                traits = self.h_traits(op, in0, in1)
                dtype_key = f"{in0},{in1}"
                blobs.append(self.h_instance(op, dtype_key, [traits]))

        return blobs

    def gen_blobs(self):
        w_p = Path(self.working_path)

        # Generate API files
        (w_p / "binary_op_api.cpp").write_text(self.content_api())
        (w_p / "binary_op_api_common.hpp").write_text(self.API_COMMON_HEADER)

        # Generate kernel instance files
        blobs = self.get_blobs()
        for b in blobs:
            (w_p / f"{b.name}.cpp").write_text(b.content)


if __name__ == "__main__":
    print('==================================')
    parser = argparse.ArgumentParser(
        prog="generate", description="Generate binary operation kernels"
    )
    parser.add_argument(
        "-w",
        "--working_path",
        default="./generated",
        help="Output directory for generated files",
    )

    parser.add_argument(
      "-o",
      "--optype",
      default="add",
      help="binary operator optype"
    )

    parser.add_argument(
      "-t",
      "--dtypes",
      default="torch.float32_torch.float32",
      help="input tensor, other tensor dtype",
    )

    args = parser.parse_args()
    print(args.working_path)

    p = Path(args.working_path)
    if not p.exists():
        p.mkdir()

    BinaryOpCodegen(args.working_path, args.dtypes, args.optype).gen_blobs()
