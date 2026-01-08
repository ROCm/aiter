#include "groupnorm.hpp"

#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <type_traits>

namespace {

template<typename T> __forceinline__ __device__ float dtype2acctype(T x) {return x;};
template<> __forceinline__ __device__ float dtype2acctype<__half>(__half x) {return __half2float(x);}
template<> __forceinline__ __device__ float dtype2acctype<__hip_bfloat16>(__hip_bfloat16 x) {return __bfloat162float(x);}

template<typename T> __forceinline__ __device__ T acctype2dtype(float x) {return x;};
template<> __forceinline__ __device__ __half acctype2dtype<__half>(float x) {return __float2half(x);}
template<> __forceinline__ __device__ __hip_bfloat16 acctype2dtype<__hip_bfloat16>(float x) {return __float2bfloat16(x);}

template<typename T,
    typename std::enable_if<sizeof(T) == sizeof(uint64_t)>::type* = nullptr>
__inline__ __device__ T warp_reduce(T value)
{
#pragma unroll
    for(uint32_t offset=(warpSize>>1); offset!=0; offset>>=1) {
        uint64_t value_u64 = *reinterpret_cast<uint64_t*>(&value);
        value_u64 = __shfl_down(value_u64, offset);
        value += *reinterpret_cast<T*>(&value_u64);
    }
    return value;
}

template<typename T, uint32_t NUM_WARPS>
__inline__ __device__ T block_reduce(T value, T identity_element, T *smem)
{
    uint32_t warp_id = threadIdx.x / warpSize;
    uint32_t lane_id = threadIdx.x % warpSize;
    value = warp_reduce(value);
    //__syncthreads();
    if(lane_id == 0) {
        smem[warp_id] = value;
    }
    __syncthreads();
    value = (threadIdx.x < NUM_WARPS) ? smem[lane_id] : identity_element;
    if(warp_id == 0) {
        value = warp_reduce(value);
    }
    return value;
}

struct Element final
{
    float mean;
    float var;
    __device__ Element & operator += (const Element & other) {
        mean += other.mean;
        var += other.var;
        return *this;
    }
};

template<typename T, uint32_t N>
struct Vec
{
    T value[N];
    __forceinline__ __device__ void load(const void *src) {
        *this = *reinterpret_cast<const Vec<T, N>*>(src);
    }
    __forceinline__ __device__ void store(void *dst) const {
        *reinterpret_cast<Vec<T, N>*>(dst) = *this;
    }
};

template<typename T, uint32_t THREADS_PER_BLOCK, uint32_t warpSize>
__device__ void groupnorm_kernel_up(uint32_t num_groups, uint32_t num_channels, int64_t numel_per_channel, 
    bool align4, const T *x, float *mean_acc, float *square_mean_acc)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t inner_size = numel_per_channel * num_channels / num_groups;

    Element el{0.0f, 0.0f};

    if(align4) {
        // the NVIDIA GPU arch can do a vectorized load of 128bits several years ago
        // i'm not sure about the latest data, but loading 64 bits at a time should be sufficient
        Vec<T, 4> vec;
        for(uint32_t i = tid*4; i < inner_size; i += (gridDim.x * THREADS_PER_BLOCK)*4) {
            uint32_t idx = blockIdx.y * inner_size + i;
            vec.load(x+idx);

            float value = dtype2acctype(vec.value[0]);
            el.mean += value;
            el.var += value * value;

            value = dtype2acctype(vec.value[1]);
            el.mean += value;
            el.var += value * value;

            value = dtype2acctype(vec.value[2]);
            el.mean += value;
            el.var += value * value;

            value = dtype2acctype(vec.value[3]);
            el.mean += value;
            el.var += value * value;
        }
    }
    else {
        for(uint32_t i = tid; i < inner_size; i += (gridDim.x * THREADS_PER_BLOCK)) {
            uint32_t idx = blockIdx.y * inner_size + i;
            float value = dtype2acctype(x[idx]);
            el.mean += value;
            el.var += value * value;
        }
    }

    static_assert(THREADS_PER_BLOCK % warpSize == 0, "");
    constexpr uint32_t NUM_WARPS = THREADS_PER_BLOCK / warpSize;
    __shared__ Element smem[NUM_WARPS];
    el = block_reduce<Element, NUM_WARPS>(el, Element{0.0f, 0.0f}, smem);

    if (threadIdx.x == 0) {
        mean_acc[blockIdx.y*gridDim.x+blockIdx.x] = el.mean;
        square_mean_acc[blockIdx.y*gridDim.x+blockIdx.x] = el.var;
    }
}

template<typename T, uint32_t THREADS_PER_BLOCK, uint32_t warpSize>
__device__ void groupnorm_kernel_down(uint32_t num_groups, uint32_t num_channels,
    int64_t numel_per_channel, float epsilon, bool align4,
    const T *x, const T *weights, const T *bias,
    const float *mean_acc, const float *square_mean_acc,
    T *y)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t inner_size = numel_per_channel * num_channels / num_groups;

    Element el{0.0f, 0.0f};
    for(uint32_t i = threadIdx.x; i < gridDim.x; i += THREADS_PER_BLOCK) {
        uint32_t idx = blockIdx.y * gridDim.x + i;
        el.mean += mean_acc[idx];
        el.var += square_mean_acc[idx];
    }

    static_assert(THREADS_PER_BLOCK % warpSize == 0, "");
    constexpr uint32_t NUM_WARPS = THREADS_PER_BLOCK / warpSize;
    __shared__ Element smem[NUM_WARPS];
    el = block_reduce<Element, NUM_WARPS>(el, Element{0.0f, 0.0f}, smem);
    if(threadIdx.x == 0) {
        smem[0] = el;
    }
    __syncthreads();
    float mean = smem[0].mean / inner_size;
    float var = smem[0].var / inner_size - mean * mean;
    float rstd = rsqrt(var + epsilon);

    if(align4) {
        Vec<float, 4> vec_x;
        float weights_value{1.0f}, bias_value{0.0f};
        Vec<T, 4> tmp;
        for(uint32_t i = tid*4; i < inner_size; i += (gridDim.x * THREADS_PER_BLOCK)*4)
        {
            uint32_t idx = blockIdx.y * inner_size + i;
            uint32_t channel_idx = (idx / numel_per_channel) % num_channels;
            tmp.load(x+idx);
            vec_x.value[0] = dtype2acctype(tmp.value[0]);
            vec_x.value[1] = dtype2acctype(tmp.value[1]);
            vec_x.value[2] = dtype2acctype(tmp.value[2]);
            vec_x.value[3] = dtype2acctype(tmp.value[3]);
            if(weights != nullptr) {
                weights_value = dtype2acctype(weights[channel_idx]);
            }
            if(bias != nullptr) {
                bias_value = dtype2acctype(bias[channel_idx]);
            }

            float value0 = (vec_x.value[0] - mean) * rstd * weights_value + bias_value;
            float value1 = (vec_x.value[1] - mean) * rstd * weights_value + bias_value;
            float value2 = (vec_x.value[2] - mean) * rstd * weights_value + bias_value;
            float value3 = (vec_x.value[3] - mean) * rstd * weights_value + bias_value;
            Vec<T, 4> vec_y{
                acctype2dtype<T>(value0), acctype2dtype<T>(value1),
                acctype2dtype<T>(value2), acctype2dtype<T>(value3)
            };
            vec_y.store(y+idx);
        }
    }
    else {
        for(uint32_t i = tid; i < inner_size; i += (gridDim.x * THREADS_PER_BLOCK))
        {
            uint32_t idx = blockIdx.y * inner_size + i;
            uint32_t channel_idx = (idx / numel_per_channel) % num_channels;
            float weight_value = (weights == nullptr) ? 1.0f : dtype2acctype(weights[channel_idx]);
            float bias_value = (bias == nullptr) ? 0.0f : dtype2acctype(bias[channel_idx]);
            float value = (dtype2acctype(x[idx]) - mean) * rstd * weight_value + bias_value;
            y[idx] = acctype2dtype<T>(value);
        }
    }
}

template<typename T, uint32_t THREADS_PER_BLOCK>
__global__ void groupnorm_kernel_dispatch_up(uint32_t num_groups, uint32_t num_channels, int64_t numel_per_channel,
    bool align4, const T *x, float *mean_acc, float *square_mean_acc) {
    if (warpSize == 32) {
        groupnorm_kernel_up<T, THREADS_PER_BLOCK, 32>(
            num_groups,
            num_channels,
            numel_per_channel,
            align4,
            x,
            mean_acc,
            square_mean_acc);
    } else if (warpSize == 64) {
        groupnorm_kernel_up<T, THREADS_PER_BLOCK, 64>(
            num_groups,
            num_channels,
            numel_per_channel,
            align4,
            x,
            mean_acc,
            square_mean_acc);
    } else {
        uint32_t size = warpSize;
        printf("Error: Unsupported warpSize = %d. Only 32 and 64 are supported.\n", size);
        assert(false);
    }
}

template<typename T, uint32_t THREADS_PER_BLOCK>
__global__ void groupnorm_kernel_dispatch_down(uint32_t num_groups, uint32_t num_channels,
    int64_t numel_per_channel, float epsilon, bool align4,
    const T *x, const T *weights, const T *bias,
    const float *mean_acc, const float *square_mean_acc,
    T *y) {
    if (warpSize == 32) {
        groupnorm_kernel_down<T, THREADS_PER_BLOCK, 32>(
            num_groups,
            num_channels,
            numel_per_channel,
            epsilon,
            align4,
            x,
            weights,
            bias,
            mean_acc,
            square_mean_acc,
            y);
    } else if (warpSize == 64) {
        groupnorm_kernel_down<T, THREADS_PER_BLOCK, 64>(
            num_groups,
            num_channels,
            numel_per_channel,
            epsilon,
            align4,
            x,
            weights,
            bias,
            mean_acc,
            square_mean_acc,
            y);
    } else {
        uint32_t size = warpSize;
        printf("Error: Unsupported warpSize = %d. Only 32 and 64 are supported.\n", size);
        assert(false);
    }
}

} // namespace

namespace rocm_torch_x {

template<typename T>
torch::Tensor GroupNorm::launchGroupNormKernel(uint32_t num_groups, float epsilon,
    const torch::Tensor x, const torch::Tensor weights, const torch::Tensor bias, hipStream_t stream)
{
    torch::Tensor y = torch::empty_like(x);
    const std::vector<int64_t> & dims = x.sizes().vec();
    int64_t numel = std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<int64_t>());
    int64_t numel_per_channel = numel / dims[0] / dims[1];
    uint32_t num_channels = dims[1];

    uint32_t outer_size = dims[0] * num_groups;
    int64_t inner_size = numel / outer_size;

    constexpr uint32_t THREADS_PER_BLOCK = 1024;
    constexpr uint32_t STEPS_PER_THREAD = 8;
    uint32_t gridx = (inner_size + (STEPS_PER_THREAD * THREADS_PER_BLOCK) - 1) / (STEPS_PER_THREAD * THREADS_PER_BLOCK);

    bool align4 = false;
    if(inner_size % 4 == 0 && gridx >= 16) {
        gridx = std::max<uint32_t>(1, gridx / 4);
        align4 = true;
    }
    gridx = std::min<uint32_t>((4096+outer_size-1)/outer_size, gridx);

    const dim3 grid_dim(gridx, outer_size, 1);
    constexpr dim3 block_dim(THREADS_PER_BLOCK, 1, 1);

    uint32_t num_acc_slots = gridx * outer_size;
    reserveMeanAccumulator(num_acc_slots*2, x.device());

    // there are some other ways:
    //    1) use sequential atomicAdd in the first function to reduce, this may influence precision, and need an another memset kenrel
    //    2) use cooperative groups to sync grid, results in hipErrorCooperativeLaunchTooLarge
    // so i launch 2 differnt kernels(up && down)
    // in fact, the second kernel is not needed if gridx==1
    // but this is definitely a case with a small amount of data, so the overall difference seems minimal.
    groupnorm_kernel_dispatch_up<T, THREADS_PER_BLOCK><<<grid_dim, block_dim, 0, stream>>>(
        num_groups,
        num_channels,
        numel_per_channel,
        align4,
        static_cast<const T*>(x.data_ptr()),
        mean_accumulator_.mutable_data_ptr<float>(),
        mean_accumulator_.mutable_data_ptr<float>()+num_acc_slots);
    HIP_CALL(hipGetLastError());

    groupnorm_kernel_dispatch_down<T, THREADS_PER_BLOCK><<<grid_dim, block_dim, 0, stream>>>(
        num_groups,
        num_channels,
        numel_per_channel,
        epsilon,
        align4,
        static_cast<const T*>(x.data_ptr()),
        static_cast<const T*>(weights.data_ptr()),
        static_cast<const T*>(bias.data_ptr()),
        mean_accumulator_.data_ptr<float>(),
        mean_accumulator_.data_ptr<float>()+num_acc_slots,
        static_cast<T*>(y.mutable_data_ptr()));
    HIP_CALL(hipGetLastError());

    return y;
}

std::optional<torch::Tensor> GroupNorm::Run(
    torch::Tensor x,
    int num_groups,
    torch::Tensor weights,
    torch::Tensor bias,
    float epsilon)
{
    at::DeviceGuard device_guard(x.device());

    auto hip_stream = c10::hip::getCurrentHIPStream();

    if (x.requires_grad()) {
        return std::nullopt;
    }

    if(weights.numel() == 0 || bias.numel() == 0) {
        return std::nullopt;
    }

    // TODO(limou) :
    if(!x.is_contiguous()) {
        x = x.contiguous();
    }
    weights = weights.contiguous();
    bias = bias.contiguous();

    torch::Tensor y;
    switch(x.scalar_type()) {
        case c10::ScalarType::Float:
            y = launchGroupNormKernel<float>(num_groups, epsilon, x, weights, bias, hip_stream.stream());
            break;
        case c10::ScalarType::Half:
            y = launchGroupNormKernel<__half>(num_groups, epsilon, x, weights, bias, hip_stream.stream());
            break;
        case c10::ScalarType::BFloat16:
            y = launchGroupNormKernel<__hip_bfloat16>(num_groups, epsilon, x, weights, bias, hip_stream.stream());
            break;
        default:
            return std::nullopt;
    }
    return y;
}

void GroupNorm::reserveMeanAccumulator(uint32_t nums_to_reserve, torch::Device device)
{
    if(nums_to_reserve <= mean_accumulator_.numel()) {
        return;
    }
    auto options = torch::TensorOptions()
        .dtype(c10::ScalarType::Float)
        .device(device)
        .requires_grad(false);

    mean_accumulator_ = at::empty({nums_to_reserve}, options);
}

} // rocm_torch_x

