#include <torch/extension.h>
#include "../include/groupnorm.hpp"

torch::Tensor _groupnorm_run_wrapper(
    torch::Tensor input,
    int64_t num_groups,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    rocm_torch_x::GroupNorm gn;
    auto result = gn.Run(
        input,
        static_cast<int>(num_groups),
        weight,
        bias,
        static_cast<float>(eps)
    );
    TORCH_CHECK(result.has_value(), "GroupNorm kernel returned nullopt");
    return result.value();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_groupnorm_run", &_groupnorm_run_wrapper);  
}
