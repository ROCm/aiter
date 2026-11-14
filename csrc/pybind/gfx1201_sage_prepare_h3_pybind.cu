// SPDX-License-Identifier: Apache-2.0
// Binding for the third-party H3 prepare; see kernels/gfx1201_sage_prepare_h3/NOTICE.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> prepare_key_int8_nhd_gfx12(torch::Tensor key, torch::Tensor mean);
std::vector<torch::Tensor> prepare_value_fp8_nhd_fused_gfx12(torch::Tensor key, torch::Tensor value);

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    m.def("gfx1201_sage_prepare_h3_key", &prepare_key_int8_nhd_gfx12);
    m.def("gfx1201_sage_prepare_h3_value", &prepare_value_fp8_nhd_fused_gfx12);
}
