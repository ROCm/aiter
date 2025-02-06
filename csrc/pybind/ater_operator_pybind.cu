#include "ater_operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &ater_add, "apply for add with transpose and broadcast.");
    m.def("mul", &ater_mul, "apply for mul with transpose and broadcast.");
    m.def("sub", &ater_sub, "apply for sub with transpose and broadcast.");
    m.def("div", &ater_div, "apply for div with transpose and broadcast.");
    m.def("tanh", &ater_tanh, "apply for tanh.");
    m.def("sigmoid", &ater_sigmoid, "apply for sigmoid.");
}