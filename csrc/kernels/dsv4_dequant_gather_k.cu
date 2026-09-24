// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Gather + dequantize of the DeepSeek V4 paged K record.
//
// Record layout inside one paged block (all bytes, k_cache is uint8):
//   [0,  block_size * 576)                 token data, 576 B per token
//        token: [0, 448)   fp8 e4m3 NoPE dims (one value per dim)
//               [448, 576) 64 bf16 RoPE dims
//   [block_size * 576, ...)                UE8M0 scales, 8 B per token,
//                                          7 of which are used (64 dims each)
//
// One wavefront handles one gathered token. Lane L owns output elements
// [L*8, L*8+8), so the 64 lanes of a wave cover all 512 dims and the store is
// a single fully coalesced 1024 B dwordx4 line. Lanes 0..55 read the fp8
// payload (8 B each, one dwordx2), lanes 56..63 read the bf16 tail (16 B each,
// one dwordx4) and pass it through unchanged.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "dsv4_dequant_gather_k.h"

namespace aiter {

namespace {

constexpr int kFp8Dim        = 448; // fp8 dims, also the fp8 payload in bytes
constexpr int kOutDim        = 512; // dims written per token
constexpr int kTokenDataSize = 576; // bytes of token data (448 + 64 * 2)
constexpr int kScaleDim      = 8;   // scale bytes per token (7 used + pad)
constexpr int kQuantBlock    = 64;  // dims per UE8M0 scale
constexpr int kWave          = 64;
constexpr int kWavesPerBlock = 4;
constexpr int kBlockThreads  = kWave * kWavesPerBlock;
constexpr int kElemsPerLane  = kOutDim / kWave;         // 8
constexpr int kFp8Lanes      = kFp8Dim / kElemsPerLane; // 56

// Branchless e4m3 -> fp32. `bias` is 7 for OCP e4m3 and 8 for e4m3fnuz;
// `denorm_scale` must be 2^(1 - bias - 3), the value of a mantissa ulp in the
// subnormal range. NaN encodings (0x80 fnuz, 0x7f/0xff OCP) are not
// represented: the V4 encoder clamps to FP8_MAX so they never occur in a cache
// record, and the branchless form is worth more than reproducing them.
__device__ __forceinline__ float fp8_e4m3_to_f32(uint32_t byte, uint32_t bias, float denorm_scale)
{
    const uint32_t sign = (byte & 0x80u) << 24;
    const uint32_t rest = byte & 0x7fu;
    // Normal: exponent field lands at bits 23..26, mantissa at bits 20..22.
    const float normal = __uint_as_float((rest << 20) + ((127u - bias) << 23));
    // exp == 0: the mantissa is the whole magnitude, in units of denorm_scale.
    const float sub = static_cast<float>(rest) * denorm_scale;
    const float mag = (rest < 8u) ? sub : normal;
    return __uint_as_float(__float_as_uint(mag) | sign);
}

// Round-to-nearest-even fp32 -> bf16, matching Triton's `.to(tl.bfloat16)`.
__device__ __forceinline__ uint32_t f32_to_bf16_rne(float f)
{
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__global__ __launch_bounds__(kBlockThreads) void dsv4_dequant_gather_k_kernel(
    uint16_t* __restrict__ out,
    const int64_t out_stride0,
    const int64_t out_stride1,
    const uint8_t* __restrict__ k_cache,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ gather_lens,
    const int32_t* __restrict__ block_table,
    const int32_t max_blocks_per_seq,
    const int32_t cache_block_size,
    const int64_t block_stride,
    const int32_t offset,
    const uint32_t fp8_bias,
    const float denorm_scale)
{
    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x & (kWave - 1);
    const int wave      = threadIdx.x >> 6;

    const int32_t seq_len    = seq_lens[batch_idx];
    const int32_t gather_len = (gather_lens == nullptr) ? seq_len : gather_lens[batch_idx];
    const int32_t start_pos  = seq_len - gather_len;

    const int32_t* __restrict__ bt_row =
        block_table + static_cast<int64_t>(batch_idx) * max_blocks_per_seq;
    uint16_t* __restrict__ out_base = out + static_cast<int64_t>(batch_idx) * out_stride0;

    // Byte offset of this lane's slice inside the token record, and the index
    // of the UE8M0 scale that covers it (unused by the bf16 tail lanes).
    const bool fp8_lane    = lane < kFp8Lanes;
    const int src_byte_off = fp8_lane ? (lane * kElemsPerLane) : (lane * 16 - kFp8Dim);
    const int scale_idx    = (lane * kElemsPerLane) / kQuantBlock;

    const int64_t token_stride = static_cast<int64_t>(gridDim.y) * kWavesPerBlock;
    int64_t i                  = static_cast<int64_t>(blockIdx.y) * kWavesPerBlock + wave;

    for(; i < gather_len; i += token_stride)
    {
        const int32_t pos          = start_pos + static_cast<int32_t>(i);
        const int32_t block_in_seq = pos / cache_block_size;
        const int32_t pos_in_block = pos - block_in_seq * cache_block_size;

        // int64: physical_block_idx * block_stride overflows int32 once the
        // cache holds enough blocks (the Triton reference widens here too).
        const uint8_t* __restrict__ blk =
            k_cache + static_cast<int64_t>(bt_row[block_in_seq]) * block_stride;
        const uint8_t* __restrict__ tok = blk + pos_in_block * kTokenDataSize;

        uint4 packed;
        if(fp8_lane)
        {
            const uint8_t* __restrict__ scl =
                blk + cache_block_size * kTokenDataSize + pos_in_block * kScaleDim;
            const float scale = exp2f(static_cast<float>(scl[scale_idx]) - 127.0f);

            const uint2 raw     = *reinterpret_cast<const uint2*>(tok + src_byte_off);
            const uint32_t w[2] = {raw.x, raw.y};
            uint32_t packed_w[4];
#pragma unroll
            for(int k = 0; k < kElemsPerLane; ++k)
            {
                const uint32_t b = (w[k >> 2] >> ((k & 3) * 8)) & 0xffu;
                const uint32_t h =
                    f32_to_bf16_rne(fp8_e4m3_to_f32(b, fp8_bias, denorm_scale) * scale);
                if((k & 1) == 0)
                    packed_w[k >> 1] = h;
                else
                    packed_w[k >> 1] |= h << 16;
            }
            packed = make_uint4(packed_w[0], packed_w[1], packed_w[2], packed_w[3]);
        }
        else
        {
            // bf16 RoPE tail: copied through byte for byte.
            packed = *reinterpret_cast<const uint4*>(tok + src_byte_off);
        }

        uint16_t* __restrict__ out_row = out_base + (offset + i) * out_stride1;
        *reinterpret_cast<uint4*>(out_row + lane * kElemsPerLane) = packed;
    }
}

} // namespace

void dsv4_dequantize_and_gather_k(aiter_tensor_t& out,
                                  const aiter_tensor_t& k_cache,
                                  const aiter_tensor_t& seq_lens,
                                  std::optional<aiter_tensor_t> gather_lens,
                                  const aiter_tensor_t& block_table,
                                  const int64_t block_size,
                                  const int64_t offset,
                                  const bool use_fnuz)
{
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_u8 || k_cache.dtype() == AITER_DTYPE_i8,
                "k_cache must be a byte tensor");
    AITER_CHECK(k_cache.dim() == 3 && k_cache.size(-1) == 584,
                "k_cache must be [num_blocks, block_size, 584]");
    AITER_CHECK(seq_lens.dtype() == AITER_DTYPE_i32, "seq_lens must be int32");
    AITER_CHECK(block_table.dtype() == AITER_DTYPE_i32, "block_table must be int32");
    AITER_CHECK(out.dim() == 3 && out.size(-1) >= kOutDim,
                "out must be [num_reqs, max_num_tokens, head_size>=512]");
    AITER_CHECK(out.stride(-1) == 1, "out last dim must be contiguous");

    const int64_t num_reqs = seq_lens.size(0);
    AITER_CHECK(num_reqs > 0, "seq_lens must be non-empty");
    AITER_CHECK(out.size(0) >= num_reqs, "out must have one row per request");
    AITER_CHECK(block_table.dim() == 2 && block_table.size(0) >= num_reqs,
                "block_table must be [num_reqs, max_blocks_per_seq]");

    const int64_t block_stride = k_cache.stride(0);
    // Lane slices are 8 B (fp8) / 16 B (bf16 tail) wide and the stores are
    // 16 B, so every base address the kernel forms must stay 16 B aligned.
    AITER_CHECK(block_stride % 16 == 0, "k_cache block stride must be 16 B aligned");
    AITER_CHECK(k_cache.stride(1) == 584 && k_cache.stride(2) == 1,
                "k_cache must be contiguous within a block");
    AITER_CHECK(block_size * 584 <= block_stride,
                "k_cache block stride is too small for block_size");
    AITER_CHECK(out.stride(1) % 8 == 0 && out.stride(0) % 8 == 0,
                "out strides must be a multiple of 8 elements");

    const int32_t* gather_lens_ptr = nullptr;
    if(gather_lens.has_value())
    {
        AITER_CHECK(gather_lens->dtype() == AITER_DTYPE_i32, "gather_lens must be int32");
        AITER_CHECK(gather_lens->size(0) >= num_reqs,
                    "gather_lens must have one entry per request");
        gather_lens_ptr = reinterpret_cast<const int32_t*>(gather_lens->data_ptr());
    }

    HipDeviceGuard device_guard(out.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // Enough waves to fill the device without making the residual tail long.
    const int64_t target_blocks = static_cast<int64_t>(get_num_cu_func()) * 8;
    int64_t blocks_y            = (target_blocks + num_reqs - 1) / num_reqs;
    blocks_y                    = blocks_y < 1 ? 1 : (blocks_y > 4096 ? 4096 : blocks_y);

    const uint32_t fp8_bias  = use_fnuz ? 8u : 7u;
    const float denorm_scale = exp2f(1.0f - static_cast<float>(fp8_bias) - 3.0f);

    const dim3 grid(static_cast<uint32_t>(num_reqs), static_cast<uint32_t>(blocks_y));
    dsv4_dequant_gather_k_kernel<<<grid, dim3(kBlockThreads), 0, stream>>>(
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        out.stride(0),
        out.stride(1),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        reinterpret_cast<const int32_t*>(seq_lens.data_ptr()),
        gather_lens_ptr,
        reinterpret_cast<const int32_t*>(block_table.data_ptr()),
        static_cast<int32_t>(block_table.stride(0)),
        static_cast<int32_t>(block_size),
        block_stride,
        static_cast<int32_t>(offset),
        fp8_bias,
        denorm_scale);
}

} // namespace aiter
