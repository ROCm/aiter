#include "ater_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ater_add", &ater_add, "apply for add with transpose and broadcast.");
    m.def("ater_mul", &ater_mul, "apply for mul with transpose and broadcast.");
    m.def("ater_sub", &ater_sub, "apply for sub with transpose and broadcast.");
    m.def("ater_div", &ater_div, "apply for div with transpose and broadcast.");
}