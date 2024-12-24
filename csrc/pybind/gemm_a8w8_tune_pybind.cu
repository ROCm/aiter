#include "gemm_a8w8.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8_tune", &gemm_a8w8_tune, "gemm_a8w8_tune", py::arg("XQ"), py::arg("WQ"),
          py::arg("x_scale"), py::arg("w_scale"), py::arg("Out"), py::arg("kernelId") = 0,
          py::arg("splitK") = 0);
}
