#include "rocm_ops.hpp"
#include "ck_tile_gemm_bf16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    CK_TILE_GEMM_BF16_PYBIND;
}
