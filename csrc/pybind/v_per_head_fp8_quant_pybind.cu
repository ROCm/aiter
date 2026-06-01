#include <torch/extension.h>

namespace aiter {
std::tuple<at::Tensor, at::Tensor> v_2way_per_head_fp8_quant(at::Tensor& v0,
                                                             at::Tensor& v1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v_2way_per_head_fp8_quant",
          &aiter::v_2way_per_head_fp8_quant,
          py::arg("v0"),
          py::arg("v1"));
}
