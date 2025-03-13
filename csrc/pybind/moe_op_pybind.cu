/* SPDX-License-Identifier: MIT
   Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "moe_op.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("topk_softmax", &topk_softmax,
            "Apply topk softmax to the gating outputs.");
      m.def("grouped_topk", &grouped_topk,
            py::arg("gating_output"), 
            py::arg("topk_weights"), py::arg("topk_ids"),
            py::arg("num_expert_group"), py::arg("topk_grp"),
            py::arg("need_renorm"), py::arg("scoring_func") = "softmax",
            py::arg("routed_scaling_factor") = 1.0f,
            "Apply grouped topk softmax/sigmodd to the gating outputs.");
      m.def("biased_grouped_topk", &biased_grouped_topk,
            py::arg("gating_output"), py::arg("correction_bias"),
            py::arg("topk_weights"), py::arg("topk_ids"),
            py::arg("num_expert_group"), py::arg("topk_grp"),
            py::arg("need_renorm"), 
            py::arg("routed_scaling_factor") = 1.0f,
            "Apply biased grouped topk softmax to the gating outputs.");
      m.def("moe_align_block_size", &moe_align_block_size,
            "Aligning the number of tokens to be processed by each expert such "
            "that it is divisible by the block size.");
      m.def("fmoe", &fmoe);
      #define EV(name) value(#name, ActivationType::name)
      py::enum_<ActivationType>(m, "ActivationType")
            .EV(Silu)
            .EV(Gelu)
            .export_values();
      #undef EV
      m.def("fmoe_int8_g1u0", &fmoe_int8_g1u0,
            py::arg("out"), py::arg("input"),
            py::arg("gate"), py::arg("down"),
            py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
            py::arg("sorted_expert_ids"), py::arg("num_valid_ids"),
            py::arg("topk"), py::arg("input_scale"),
            py::arg("fc1_scale"), py::arg("fc2_scale"),
            py::arg("fc2_smooth_scale") = std::nullopt,
            py::arg("activation") = ActivationType::Silu);
      m.def("fmoe_g1u1", &fmoe_g1u1,
            py::arg("out"), py::arg("input"),
            py::arg("gate"), py::arg("down"),
            py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
            py::arg("sorted_expert_ids"), py::arg("num_valid_ids"),
            py::arg("topk"), py::arg("input_scale"),
            py::arg("fc1_scale"), py::arg("fc2_scale"),
            py::arg("fc2_smooth_scale") = std::nullopt,
            py::arg("activation") = ActivationType::Silu);
      m.def("fmoe_int8_g1u0_a16", &fmoe_int8_g1u0_a16);
      m.def("fmoe_fp8_g1u1_a16", &fmoe_fp8_g1u1_a16);
      m.def("fmoe_fp8_blockscale_g1u1", &fmoe_fp8_blockscale_g1u1,
            py::arg("out"), py::arg("input"),
            py::arg("gate"), py::arg("down"),
            py::arg("sorted_token_ids"), py::arg("sorted_weight_buf"),
            py::arg("sorted_expert_ids"), py::arg("num_valid_ids"),
            py::arg("topk"),
            py::arg("fc1_scale"), py::arg("fc2_scale"),
            py::arg("input_scale"),
            py::arg("fc_scale_blkn") = 128, py::arg("fc_scale_blkk") = 128,
            py::arg("fc2_smooth_scale") = std::nullopt);
      m.def("moe_sum", &moe_sum, "moe_sum(Tensor! input, Tensor output) -> ()");
}