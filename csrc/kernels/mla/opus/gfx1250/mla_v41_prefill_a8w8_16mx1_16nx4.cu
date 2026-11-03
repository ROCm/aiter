#include <hip/hip_runtime.h>

#include "mla_v41_prefill_a8w8_16mx1_16nx4.hpp"

using opus_mla_v41_prefill_a8w8_16mx1_16nx4_t =
    opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits<16, 64, 4, fp8_t, bf16_t, bf16_t>;

extern "C" __global__ __launch_bounds__(opus_mla_v41_prefill_a8w8_16mx1_16nx4_t::BLOCK_SIZE, 2) void
opus_mla_v41_prefill_a8w8_16mx1_16nx4_kernel(opus_mla_v41_prefill_kargs kargs)
{
    opus_mla_v41_prefill_a8w8_16mx1_16nx4_fwd<opus_mla_v41_prefill_a8w8_16mx1_16nx4_t>(kargs);
}

extern "C" __global__ __launch_bounds__(opus_mla_v41_prefill_a8w8_16mx1_16nx4_t::BLOCK_SIZE, 2) void
opus_mla_v41_prefill_a8w8_16mx1_16nx4_split_kernel(opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs> kargs)
{
    opus_mla_v41_prefill_a8w8_16mx1_16nx4_fwd<opus_mla_v41_prefill_a8w8_16mx1_16nx4_t>(kargs);
}
