#include <torch/extension.h>

#include "aiter_hip_common.h"
#include <optional>

namespace rocm_torch_x {

class __attribute__ ((visibility("hidden"))) GroupNorm final
{
public:
    explicit GroupNorm() = default;
    ~GroupNorm() = default;
public:
    // return empty if not supported
    std::optional<torch::Tensor> Run(
        torch::Tensor x,
        int num_groups,
        torch::Tensor weights,
        torch::Tensor bias,
        float epsilon);
private:
    template<typename T>
    torch::Tensor launchGroupNormKernel(uint32_t num_groups, float epsilon,
        const torch::Tensor x, const torch::Tensor weights, const torch::Tensor bias, hipStream_t stream);

    void reserveMeanAccumulator(uint32_t nums_to_reserve, torch::Device device);
private:
    torch::Tensor mean_accumulator_;
};

} // namespace rocm_torch_x
