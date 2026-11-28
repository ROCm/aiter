#include <hip/hip_runtime.h>

#include "mla_v4_prefill_a8w8_16mx4_64nx1.hpp"

using opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_t =
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, 1, fp8_t, bf16_t, bf16_t>;

extern "C" __global__ __launch_bounds__(opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_t::BLOCK_SIZE, 1) void
opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_kernel(opus_mla_v4_prefill_fp8_kargs kargs)
{
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_t>(kargs);
}

extern "C" __global__ __launch_bounds__(opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_t::BLOCK_SIZE, 1) void
opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_split_kernel(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs> kargs)
{
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy1_t>(kargs);
}

using opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_t =
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits<16, 64, 4, 2, fp8_t, bf16_t, bf16_t>;

extern "C" __global__ __launch_bounds__(opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_t::BLOCK_SIZE, 1) void
opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_kernel(opus_mla_v4_prefill_fp8_kargs kargs)
{
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_t>(kargs);
}

extern "C" __global__ __launch_bounds__(opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_t::BLOCK_SIZE, 1) void
opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_split_kernel(opus_mla_prefill_split_kargs<opus_mla_v4_prefill_fp8_kargs> kargs)
{
    opus_mla_v4_prefill_a8w8_16mx4_64nx1_fwd<opus_mla_v4_prefill_a8w8_16mx4_64nx1_cy2_t>(kargs);
}
