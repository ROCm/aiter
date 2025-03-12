// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "activation.h"
#include "attention.h"
#include "attention_ragged.h"
#include "attention_ck.h"
#include "attention_asm.h"
#include "attention_asm_mla.h"
#include "cache.h"
#include "custom_all_reduce.h"
#include "communication_asm.h"
#include "gemm_a8w8_blockscale.h"
#include "custom.h"
#include "moe_op.h"
#include "moe_sorting.h"
#include "norm.h"
#include "pos_encoding.h"
#include "rmsnorm.h"
#include "smoothquant.h"
#include "aiter_operator.h"
#include "asm_gemm_a8w8.h"
#include <torch/extension.h>
#include "gemm_a8w8.h"
#include "batched_gemm_a8w8.h"
#include "quant.h"
#include "moe_ck.h"
#include "rope.h"
#include "rocsolgemm.cuh"
#include "hipbsolgemm.cuh"

// #include "mha_varlen_fwd.h"
// #include "mha_varlen_bwd.h"
// #include "mha_bwd.h"
// #include "mha_fwd.h"
#include "rocm_ops.hpp"

#ifdef PREBUILD_KERNELS
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      // remove *TUNE* , MHA*
      // GEMM_A8W8_TUNE_PYBIND;
      RMSNORM_PYBIND;
      // MHA_VARLEN_BWD_PYBIND;
      // MHA_BWD_PYBIND;
      GEMM_A8W8_PYBIND;
      CUSTOM_PYBIND;
      SMOOTHQUANT_PYBIND;
      BATCHED_GEMM_A8W8_PYBIND;
      MOE_CK_PYBIND;
      // BATCHED_GEMM_A8W8_TUNE_PYBIND;
      GEMM_A8W8_ASM_PYBIND;
      // MHA_VARLEN_FWD_PYBIND;
      // MHA_BWD_ASM_PYBIND;
      ACTIVATION_PYBIND;
      ATTENTION_ASM_MLA_PYBIND;
      ATTENTION_CK_PYBIND;
      MOE_SORTING_PYBIND;
      NORM_PYBIND;
      POS_ENCODING_PYBIND;
      ATTENTION_PYBIND;
      MOE_CK_2STAGES_PYBIND;
      QUANT_PYBIND;
      ATTENTION_ASM_PYBIND;
      // MHA_FWD_PYBIND;
      ATTENTION_RAGGED_PYBIND;
      MOE_OP_PYBIND;
      ROPE_PYBIND;
      // GEMM_A8W8_BLOCKSCALE_TUNE_PYBIND;
      GEMM_A8W8_BLOCKSCALE_PYBIND;
      AITER_OPERATOR_PYBIND;
      CUSTOM_ALL_REDUCE_PYBIND;
      CACHE_PYBIND;
      HIPBSOLGEMM_PYBIND;
      ROCSOLGEMM_PYBIND;
}
#endif