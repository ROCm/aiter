// SPDX-License-Identifier: MIT

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "q4_group64_gemv.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

constexpr int kGroup     = 64;
constexpr int kTileRows  = 32;
constexpr int kTileBytes = 1088;
constexpr int kWaves     = 8;
constexpr int kThreads   = kTileRows * kWaves;

enum class Mapping : int
{
    Auto       = 0,
    Old        = 1,
    Split2     = 2,
    Split4     = 3,
    Split8     = 4,
    Small8x8   = 5,
    Small8x16  = 6,
    Small8x32  = 7,
    Small16x16 = 8,
    Small16x32 = 9,
    Small32x32 = 10,
};

struct DispatchEntry
{
    int rows;
    int columns;
    Mapping mapping;
};

constexpr DispatchEntry kMeasuredDispatch[] = {
    {512, 3584, Mapping::Small32x32},
    {1024, 3072, Mapping::Small32x32},
    {1024, 4096, Mapping::Small32x32},
    {3072, 3072, Mapping::Split8},
    {3072, 8192, Mapping::Split8},
    {3584, 3584, Mapping::Split8},
    {3584, 18944, Mapping::Split8},
    {4096, 4096, Mapping::Split8},
    {4096, 12288, Mapping::Split8},
    {4096, 14336, Mapping::Split8},
    {8192, 3072, Mapping::Split4},
    {12288, 4096, Mapping::Split8},
    {14336, 4096, Mapping::Split8},
    {18944, 3584, Mapping::Split8},
};

Mapping select_mapping(int rows, int columns)
{
    for(const auto& entry : kMeasuredDispatch)
    {
        if(entry.rows == rows && entry.columns == columns)
        {
            return entry.mapping;
        }
    }
    return Mapping::Old;
}

std::string cached_gpu_arch(int device_id)
{
    static std::mutex mutex;
    static std::unordered_map<int, std::string> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(device_id);
    if(found != cache.end())
    {
        return found->second;
    }
    const std::string arch = get_gpu_arch();
    cache.emplace(device_id, arch);
    return arch;
}

bool cached_is_tuned_rx_9070_xt(int device_id)
{
    static std::mutex mutex;
    static std::unordered_map<int, bool> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(device_id);
    if(found != cache.end())
    {
        return found->second;
    }

    hipDeviceProp_t properties{};
    int pci_chip_id                    = 0;
    const hipError_t properties_status = hipGetDeviceProperties(&properties, device_id);
    const hipError_t chip_status =
        hipDeviceGetAttribute(&pci_chip_id, hipDeviceAttributePciChipId, device_id);
    const bool tuned = properties_status == hipSuccess && chip_status == hipSuccess &&
                       aiter::detail::q4_group64_is_tuned_rx_9070_xt(
                           pci_chip_id, properties.multiProcessorCount, properties.name);
    if(properties_status != hipSuccess || chip_status != hipSuccess)
    {
        // Identity discovery is an optimization guard. Clear a possible sticky
        // runtime error and fail closed to the conservative mapping.
        (void)hipGetLastError();
    }
    cache.emplace(device_id, tuned);
    return tuned;
}

bool ranges_overlap(const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes)
{
    const auto lhs_begin = reinterpret_cast<uintptr_t>(lhs);
    const auto rhs_begin = reinterpret_cast<uintptr_t>(rhs);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

bool experimental_runtime_enabled()
{
    // Follow the integer contract of aiter.jit.core.is_experimental_enabled.
    // Check every entry so removing the opt-in immediately disables an
    // already-loaded module. Full consumption prevents values such as "1junk"
    // from enabling a direct C++ call.
    const char* value = std::getenv("AITER_ENABLE_EXPERIMENTAL");
    if(value == nullptr)
    {
        return false;
    }
    errno             = 0;
    char* end         = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if(errno == ERANGE || end == value)
    {
        return false;
    }
    while(*end != '\0' && std::isspace(static_cast<unsigned char>(*end)) != 0)
    {
        ++end;
    }
    return *end == '\0' && parsed != 0;
}

class AiterCheckThrowGuard
{
    public:
    AiterCheckThrowGuard() : previous_(aiter_detail::g_aiter_can_throw)
    {
        aiter_detail::g_aiter_can_throw = true;
    }

    ~AiterCheckThrowGuard() { aiter_detail::g_aiter_can_throw = previous_; }

    AiterCheckThrowGuard(const AiterCheckThrowGuard&)            = delete;
    AiterCheckThrowGuard& operator=(const AiterCheckThrowGuard&) = delete;

    private:
    bool previous_;
};

#if defined(__gfx1201__)

__device__ __forceinline__ int low_nibble(uint8_t value)
{
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int high_nibble(uint8_t value)
{
    return int(static_cast<int8_t>(value)) >> 4;
}

template <bool CheckRowBounds>
__global__
__launch_bounds__(kThreads) void q4_group64_gemv_old_kernel(const uint8_t* __restrict__ weights,
                                                            const float* __restrict__ x,
                                                            float* __restrict__ out,
                                                            int row_tiles,
                                                            int columns)
{
    __shared__ float x_tile[kWaves][kGroup];
    const int lane     = static_cast<int>(threadIdx.x) & (kTileRows - 1);
    const int wave     = static_cast<int>(threadIdx.x) / kTileRows;
    const int row_tile = static_cast<int>(blockIdx.x) * kWaves + wave;
    if constexpr(CheckRowBounds)
    {
        if(row_tile >= row_tiles)
        {
            return;
        }
    }

    const int groups = columns / kGroup;
    float sum        = 0.0f;
    for(int group = 0; group < groups; ++group)
    {
        x_tile[wave][lane]             = x[group * kGroup + lane];
        x_tile[wave][lane + kTileRows] = x[group * kGroup + lane + kTileRows];
        __syncwarp();

        const uint8_t* tile =
            weights + (static_cast<int64_t>(row_tile) * groups + group) * kTileBytes;
        const float scale = __half2float(reinterpret_cast<const __half*>(tile)[lane]);
#pragma unroll
        for(int i = 0; i < kGroup / 2; ++i)
        {
            const uint8_t packed = tile[64 + i * kTileRows + lane];
            sum = fmaf(scale * float(low_nibble(packed)), x_tile[wave][2 * i], sum);
            sum = fmaf(scale * float(high_nibble(packed)), x_tile[wave][2 * i + 1], sum);
        }
        __syncwarp();
    }
    out[row_tile * kTileRows + lane] = sum;
}

template <int SplitWaves>
__global__
__launch_bounds__(kThreads) void q4_group64_gemv_split_kernel(const uint8_t* __restrict__ weights,
                                                              const float* __restrict__ x,
                                                              float* __restrict__ out,
                                                              int columns)
{
    static_assert(SplitWaves == 2 || SplitWaves == 4 || SplitWaves == 8);
    constexpr int kTilesPerBlock = kWaves / SplitWaves;
    __shared__ float partial[kWaves][kTileRows];

    const int lane          = static_cast<int>(threadIdx.x) & (kTileRows - 1);
    const int wave          = static_cast<int>(threadIdx.x) / kTileRows;
    const int split         = wave & (SplitWaves - 1);
    const int tile_in_block = wave / SplitWaves;
    const int row_tile      = static_cast<int>(blockIdx.x) * kTilesPerBlock + tile_in_block;
    const int groups        = columns / kGroup;
    float sum               = 0.0f;

    for(int group = split; group < groups; group += SplitWaves)
    {
        const uint8_t* tile =
            weights + (static_cast<int64_t>(row_tile) * groups + group) * kTileBytes;
        float partial_low  = 0.0f;
        float partial_high = 0.0f;
#pragma unroll 4
        for(int i = 0; i < kGroup / 2; ++i)
        {
            const float x_low    = x[group * kGroup + 2 * i];
            const float x_high   = x[group * kGroup + 2 * i + 1];
            const uint8_t packed = tile[64 + i * kTileRows + lane];
            partial_low          = fmaf(float(low_nibble(packed)), x_low, partial_low);
            partial_high         = fmaf(float(high_nibble(packed)), x_high, partial_high);
        }
        const float scale = __half2float(reinterpret_cast<const __half*>(tile)[lane]);
        sum               = fmaf(scale, partial_low + partial_high, sum);
    }

    partial[wave][lane] = sum;
    __syncthreads();
    if(split == 0)
    {
#pragma unroll
        for(int other = 1; other < SplitWaves; ++other)
        {
            sum += partial[wave + other][lane];
        }
        out[row_tile * kTileRows + lane] = sum;
    }
}

template <int SplitWaves, int SliceRows>
__global__ __launch_bounds__(SplitWaves* kTileRows) void q4_group64_gemv_small_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int columns)
{
    static_assert(SplitWaves == 8 || SplitWaves == 16 || SplitWaves == 32);
    static_assert(SliceRows == 8 || SliceRows == 16 || SliceRows == 32);
    constexpr int kSlicesPerTile = kTileRows / SliceRows;
    __shared__ float partial[SplitWaves][kTileRows];

    const int lane      = static_cast<int>(threadIdx.x) & (kTileRows - 1);
    const int wave      = static_cast<int>(threadIdx.x) / kTileRows;
    const int row_slice = static_cast<int>(blockIdx.x);
    const int row_tile  = row_slice / kSlicesPerTile;
    const int row_lane  = (row_slice % kSlicesPerTile) * SliceRows + lane;
    const int groups    = columns / kGroup;
    float sum           = 0.0f;

    if(lane < SliceRows)
    {
        for(int group = wave; group < groups; group += SplitWaves)
        {
            const uint8_t* tile =
                weights + (static_cast<int64_t>(row_tile) * groups + group) * kTileBytes;
            float partial_low  = 0.0f;
            float partial_high = 0.0f;
#pragma unroll 4
            for(int i = 0; i < kGroup / 2; ++i)
            {
                const float x_low    = x[group * kGroup + 2 * i];
                const float x_high   = x[group * kGroup + 2 * i + 1];
                const uint8_t packed = tile[64 + i * kTileRows + row_lane];
                partial_low          = fmaf(float(low_nibble(packed)), x_low, partial_low);
                partial_high         = fmaf(float(high_nibble(packed)), x_high, partial_high);
            }
            const float scale = __half2float(reinterpret_cast<const __half*>(tile)[row_lane]);
            sum               = fmaf(scale, partial_low + partial_high, sum);
        }
    }

    partial[wave][lane] = sum;
    __syncthreads();
    if(wave == 0 && lane < SliceRows)
    {
#pragma unroll
        for(int other = 1; other < SplitWaves; ++other)
        {
            sum += partial[other][lane];
        }
        out[row_tile * kTileRows + row_lane] = sum;
    }
}

#else

template <bool CheckRowBounds>
__global__ __launch_bounds__(kThreads) void q4_group64_gemv_old_kernel(
    const uint8_t*, const float*, float*, int, int)
{
    assert(false && "q4_group64_gemv requires gfx1201 device code");
}

template <int SplitWaves>
__global__ __launch_bounds__(kThreads) void q4_group64_gemv_split_kernel(const uint8_t*,
                                                                         const float*,
                                                                         float*,
                                                                         int)
{
    assert(false && "q4_group64_gemv requires gfx1201 device code");
}

template <int SplitWaves, int SliceRows>
__global__ __launch_bounds__(SplitWaves* kTileRows) void q4_group64_gemv_small_kernel(
    const uint8_t*, const float*, float*, int)
{
    assert(false && "q4_group64_gemv requires gfx1201 device code");
}

#endif

template <int SplitWaves>
void launch_split(
    const uint8_t* weights, const float* x, float* out, int rows, int columns, hipStream_t stream)
{
    constexpr int kTilesPerBlock = kWaves / SplitWaves;
    const int row_tiles          = rows / kTileRows;
    const dim3 grid(row_tiles / kTilesPerBlock);
    const dim3 block(kThreads);
    hipLaunchKernelGGL((q4_group64_gemv_split_kernel<SplitWaves>),
                       grid,
                       block,
                       0,
                       stream,
                       weights,
                       x,
                       out,
                       columns);
}

template <int SplitWaves, int SliceRows>
void launch_small(
    const uint8_t* weights, const float* x, float* out, int rows, int columns, hipStream_t stream)
{
    const dim3 grid(rows / SliceRows);
    const dim3 block(SplitWaves * kTileRows);
    hipLaunchKernelGGL((q4_group64_gemv_small_kernel<SplitWaves, SliceRows>),
                       grid,
                       block,
                       0,
                       stream,
                       weights,
                       x,
                       out,
                       columns);
}

} // namespace

namespace aiter {

void q4_group64_gemv_out(aiter_tensor_t& x,
                         aiter_tensor_t& packed_weight,
                         aiter_tensor_t& out,
                         int mapping_value)
{
    AiterCheckThrowGuard check_throw_guard;
    AITER_CHECK(experimental_runtime_enabled(),
                "q4_group64_gemv is experimental; set AITER_ENABLE_EXPERIMENTAL=1 before "
                "calling it");
    AITER_CHECK(x.is_gpu() && packed_weight.is_gpu() && out.is_gpu(),
                "x, packed_weight, and out must be GPU tensors");
    AITER_CHECK(x.device_id == packed_weight.device_id && x.device_id == out.device_id,
                "x, packed_weight, and out must be on the same device");
    AITER_CHECK(x.dtype() == AITER_DTYPE_fp32, "x must be FP32");
    AITER_CHECK(packed_weight.dtype() == AITER_DTYPE_u8, "packed_weight must be uint8");
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be FP32");
    AITER_CHECK(x.dim() == 1 && x.is_contiguous(), "x must be contiguous [K]");
    AITER_CHECK(packed_weight.dim() == 3 && packed_weight.is_contiguous(),
                "packed_weight must be contiguous [N/32,K/64,1088]");
    AITER_CHECK(out.dim() == 1 && out.is_contiguous(), "out must be contiguous [N]");

    const int64_t columns64 = x.size(0);
    AITER_CHECK(columns64 > 0 && columns64 % kGroup == 0, "K must be positive and divisible by 64");
    AITER_CHECK(columns64 <= std::numeric_limits<int>::max(), "K is too large");
    AITER_CHECK(packed_weight.size(0) > 0 && packed_weight.size(2) == kTileBytes,
                "packed_weight must have shape [N/32,K/64,1088]");
    AITER_CHECK(packed_weight.size(1) * kGroup == columns64,
                "packed_weight K dimension must match x");
    const int64_t rows64 = packed_weight.size(0) * kTileRows;
    AITER_CHECK(rows64 <= std::numeric_limits<int>::max(), "N is too large");
    AITER_CHECK(out.size(0) == rows64, "out shape must be [N]");

    AITER_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % alignof(float) == 0,
                "x must be naturally aligned");
    AITER_CHECK(reinterpret_cast<uintptr_t>(packed_weight.data_ptr()) % alignof(__half) == 0,
                "packed_weight must be 2-byte aligned for FP16 scales");
    AITER_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % alignof(float) == 0,
                "out must be naturally aligned");
    AITER_CHECK(!ranges_overlap(out.data_ptr(),
                                out.numel() * out.element_size(),
                                x.data_ptr(),
                                x.numel() * x.element_size()),
                "out must not overlap x");
    AITER_CHECK(!ranges_overlap(out.data_ptr(),
                                out.numel() * out.element_size(),
                                packed_weight.data_ptr(),
                                packed_weight.numel() * packed_weight.element_size()),
                "out must not overlap packed_weight");

    AITER_CHECK(mapping_value >= static_cast<int>(Mapping::Auto) &&
                    mapping_value <= static_cast<int>(Mapping::Small32x32),
                "invalid q4_group64_gemv mapping");
    const int rows    = static_cast<int>(rows64);
    const int columns = static_cast<int>(columns64);
    HipDeviceGuard device_guard(x.device_id);
    const std::string arch = cached_gpu_arch(x.device_id);
    AITER_CHECK(arch == "gfx1201", "q4_group64_gemv requires gfx1201, got ", arch);

    Mapping mapping = static_cast<Mapping>(mapping_value);
    if(mapping == Mapping::Auto)
    {
        mapping =
            cached_is_tuned_rx_9070_xt(x.device_id) ? select_mapping(rows, columns) : Mapping::Old;
    }

    if(mapping == Mapping::Split2)
    {
        AITER_CHECK(rows % 128 == 0, "split2 requires N divisible by 128");
    }
    else if(mapping == Mapping::Split4)
    {
        AITER_CHECK(rows % 64 == 0, "split4 requires N divisible by 64");
    }
    // split8 and every small mapping accept the packed layout's N % 32 contract.

    const auto* weights      = reinterpret_cast<const uint8_t*>(packed_weight.data_ptr());
    const auto* x_ptr        = reinterpret_cast<const float*>(x.data_ptr());
    auto* out_ptr            = reinterpret_cast<float*>(out.data_ptr());
    const hipStream_t stream = aiter::getCurrentHIPStream();

    switch(mapping)
    {
    case Mapping::Old: {
        const int row_tiles = rows / kTileRows;
        const dim3 grid((row_tiles + kWaves - 1) / kWaves);
        const dim3 block(kThreads);
        // Preserve the original check-free path for full eight-tile blocks,
        // while keeping the conservative fallback safe for every legal N % 32.
        if(row_tiles % kWaves == 0)
        {
            hipLaunchKernelGGL((q4_group64_gemv_old_kernel<false>),
                               grid,
                               block,
                               0,
                               stream,
                               weights,
                               x_ptr,
                               out_ptr,
                               row_tiles,
                               columns);
        }
        else
        {
            hipLaunchKernelGGL((q4_group64_gemv_old_kernel<true>),
                               grid,
                               block,
                               0,
                               stream,
                               weights,
                               x_ptr,
                               out_ptr,
                               row_tiles,
                               columns);
        }
        break;
    }
    case Mapping::Split2: launch_split<2>(weights, x_ptr, out_ptr, rows, columns, stream); break;
    case Mapping::Split4: launch_split<4>(weights, x_ptr, out_ptr, rows, columns, stream); break;
    case Mapping::Split8: launch_split<8>(weights, x_ptr, out_ptr, rows, columns, stream); break;
    case Mapping::Small8x8:
        launch_small<8, 8>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Small8x16:
        launch_small<8, 16>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Small8x32:
        launch_small<8, 32>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Small16x16:
        launch_small<16, 16>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Small16x32:
        launch_small<16, 32>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Small32x32:
        launch_small<32, 32>(weights, x_ptr, out_ptr, rows, columns, stream);
        break;
    case Mapping::Auto: AITER_CHECK(false, "unreachable auto mapping");
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
