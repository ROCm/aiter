// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "activation.h"
#include "attention.h"
#include "attention_ck.h"
#include "attention_asm.h"
#include "cache.h"
//#include "custom_all_reduce.h"
//#include "communication_asm.h"
//#include "custom.h"
#include "moe_op.h"
#include "moe_sorting.h"
//#include "norm.h"
//#include "pos_encoding.h"
//#include "rmsnorm.h"
#include "smoothquant.h"
#include "aiter_operator.h"
//#include "asm_gemm_a8w8.h"
#include <torch/extension.h>
//#include "gemm_a8w8.h"
#include "quant.h"
#include "moe_ck.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("moe_align_block_size", &moe_align_block_size,
            "Aligning the number of tokens to be processed by each expert such "
            "that it is divisible by the block size.");
      m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU.");
      m.def("moe_sum", &moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");
      m.def("paged_attention_rocm", &paged_attention,
            "paged_attention_rocm(Tensor! out, Tensor workspace_buffer,"
            "                Tensor query, Tensor key_cache,"
            "                Tensor value_cache,"
            "                float scale, Tensor kv_indptr,"
            "                Tensor kv_page_indices, Tensor kv_last_page_lens,"
            "                int block_size,"
            "                int max_num_partitions,"
            "                Tensor? alibi_slopes,"
            "                str kv_cache_dtype,"
            "                str kv_cache_layout,"
            "                float logits_soft_cap,"
            "                Tensor k_scale, Tensor v_scale) -> ()");

      m.def("swap_blocks", &swap_blocks,
            "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
      m.def("copy_blocks", &copy_blocks,
            "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
            "Tensor block_mapping) -> ()");

      m.def("reshape_and_cache", &reshape_and_cache,
            "reshape_and_cache(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_flash", &reshape_and_cache_flash,
            "reshape_and_cache_flash(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype,"
            "                        float k_scale, float v_scale) -> ()");
      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor! k_dequant_scales, Tensor! v_dequant_scales,"
            "                  Tensor slot_mapping) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");

      m.def("smoothquant_fwd", &smoothquant_fwd);
      m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);
      m.def("moe_sorting_fwd", &moe_sorting_fwd);
      m.def("pa_fwd_naive", &pa_fwd_naive, "pa_fwd_naive",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("k_dequant_scales"),
            py::arg("v_dequant_scales"),
            py::arg("max_seq_len"),
            py::arg("num_kv_heads"),
            py::arg("scale_s"),
            py::arg("scale_k"),
            py::arg("scale_v"),
            py::arg("block_size"),
            py::arg("quant_algo"),
            py::arg("out_") = std::nullopt);
      // ck staff end

      m.def("fmoe", &fmoe);
      m.def("fmoe_int8_g1u0", &fmoe_int8_g1u0);
      m.def("fmoe_g1u1", &fmoe_g1u1,
            py::arg("out"), py::arg("input"),
            py::arg("gate"), py::arg("down"),
            py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
            py::arg("sorted_expert_ids"), py::arg("num_tokens_post_padded"),
            py::arg("topk"), py::arg("input_scale"),
            py::arg("fc1_scale"), py::arg("fc2_scale"),
            py::arg("fc2_smooth_scale") = std::nullopt);
      m.def("fmoe_int8_g1u0_a16", &fmoe_int8_g1u0_a16);
      m.def("pa_fwd_asm", &pa_fwd, "pa_fwd",
            py::arg("Q"),
            py::arg("K"),
            py::arg("V"),
            py::arg("block_tables"),
            py::arg("context_lens"),
            py::arg("max_num_blocks"),
            py::arg("K_QScale") = std::nullopt,
            py::arg("V_QScale") = std::nullopt,
            py::arg("out_") = std::nullopt);

      m.def("reshape_and_cache_with_pertoken_quant", &reshape_and_cache_with_pertoken_quant,
            "reshape_and_cache_with_pertoken_quant(Tensor key, Tensor value,"
            "                        Tensor! key_cache,"
            "                        Tensor! value_cache,"
            "                        Tensor! k_dequant_scales,"
            "                        Tensor! v_dequant_scales,"
            "                        Tensor slot_mapping,"
            "                        str kv_cache_dtype) -> ()");
      m.def("convert_fp8", &convert_fp8,
            "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
            "str kv_cache_dtype) -> ()");

      m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant);
      m.def("dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant);
      m.def("dynamic_per_token_scaled_fp8_quant", &dynamic_per_token_scaled_fp8_quant,
            py::arg("out"), py::arg("input"),
            py::arg("scales"), py::arg("scale_ub") = std::nullopt);
      m.def("ck_moe", &ck_moe,
            py::arg("hidden_states"), py::arg("w1"), py::arg("w2"),
            py::arg("topk_weights"), py::arg("topk_ids"),
            py::arg("w1_scale") = std::nullopt, py::arg("w2_scale") = std::nullopt,
            py::arg("a1_scale") = std::nullopt, py::arg("a2_scale") = std::nullopt,
            py::arg("block_m") = 32);
}
