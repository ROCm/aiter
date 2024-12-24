#include "asm_gemm_a8w8.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("gemm_a8w8_asm", &gemm_a8w8_asm,
            "Asm gemm a8w8 ,  weight should be shuffle to layout(32,16)",
            py::arg("XQ"), py::arg("WQ"),
            py::arg("x_scale"), py::arg("w_scale"),
            py::arg("Out"), py::arg("bias"),
            py::arg("sub_m") = 128, py::arg("sub_n") = 128,
            py::arg("pad_a") = 0, py::arg("pad_b") = 0,
            py::arg("pad_c") = 0, py::arg("splitK") = 0);
}