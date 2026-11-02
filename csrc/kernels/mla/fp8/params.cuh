// SPDX-License-Identifier: MIT
#pragma once
#include <cmath>
#include <cstdint>
#include <hip/hip_runtime.h>
namespace sparse_mla_fp8 {
struct Params
{
    const uint8_t *q, *kv;
    const int32_t* indices;
    uint16_t* partial_o;
    float* partial_lse;
    uint8_t* out;
    float* lse;
    int queries, slots, capacity;
    float q_scale, kv_scale, attention_scale;
    static constexpr float p_scale = 1.f / 256.f;
    int64_t q_stride_row, q_stride_head;
    bool bf16_q;
};
inline int split_count(int queries)
{
    if(queries <= 6)
        return 17;
    if(queries <= 12)
        return 9;
    if(queries <= 21)
        return 6;
    if(queries <= 23)
        return 4;
    if(queries <= 42)
        return 6;
    if(queries <= 64)
        return 4;
    if(queries <= 128)
        return 2;
    return 1;
}
hipError_t launch_small(Params p, hipStream_t stream);
hipError_t launch_prefetch(Params p, hipStream_t stream);
hipError_t launch_rotating(Params p, hipStream_t stream);
hipError_t launch_streaming(Params p, hipStream_t stream);
} // namespace sparse_mla_fp8
