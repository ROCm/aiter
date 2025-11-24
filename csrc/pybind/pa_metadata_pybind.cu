#include "rocm_ops.hpp"
#include "pa.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    PA_METADATA_PYBIND;
}
