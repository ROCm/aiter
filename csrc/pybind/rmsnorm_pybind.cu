#include "rmsnorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm_cu", &rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
    m.def("fused_add_rms_norm_cu", &fused_add_rms_norm, "In-place fused Add and RMS Normalization");
    m.def("rmsnorm2d_fwd", &rmsnorm2d);
    m.def("rmsnorm2d_fwd_with_add", &rmsnorm2d_with_add);
    m.def("rmsnorm2d_fwd_with_smoothquant", &rmsnorm2d_with_smoothquant);
    m.def("rmsnorm2d_fwd_with_add_smoothquant", &rmsnorm2d_with_add_smoothquant);
    m.def("rmsnorm2d_fwd_with_dynamicquant", &rmsnorm2d_with_dynamicquant);
    m.def("rmsnorm2d_fwd_with_add_dynamicquant", &rmsnorm2d_with_add_dynamicquant);
}