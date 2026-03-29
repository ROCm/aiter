// TurboQuant KV Cache Compress/Decompress GPU Kernels (2/3/4-bit)
//
// Compress:  kv_data (T, D, bf16) -> packed buf (T, compressed_bytes, uint8)
//   Per group: L2 norm -> normalize -> rotate (Pi @ x) -> scale(sqrt(d)) -> searchsorted -> pack
//
// Decompress: packed buf -> kv_data (T, D, bf16)
//   Per group: unpack -> codebook lookup -> inv_rotate (x @ Pi) -> denormalize
//
// Template parameter BIT_WIDTH selects codebook size and packing format:
//   4-bit: 16 levels, 15 boundaries, 2 values/byte
//   3-bit: 8 levels,  7 boundaries, 8 values/3 bytes
//   2-bit: 4 levels,  3 boundaries, 4 values/byte

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Packed bytes for GROUP_DIM elements at BIT_WIDTH
template <int GROUP_DIM, int BIT_WIDTH>
struct PackedSize;
template <int G> struct PackedSize<G, 4> { static constexpr int value = G / 2; };
template <int G> struct PackedSize<G, 3> { static constexpr int value = G * 3 / 8; };
template <int G> struct PackedSize<G, 2> { static constexpr int value = G / 4; };


// ─── Compress Kernel ───
template <int GROUP_DIM, int BIT_WIDTH>
__global__ void tq_kv_compress_kernel(
    const __hip_bfloat16* __restrict__ kv_data,
    const float* __restrict__ Pi_all,
    const float* __restrict__ boundaries,
    uint8_t* __restrict__ output,
    int T, int kv_lora_rank, int qk_rope_head_dim,
    int n_groups, int compressed_bytes)
{
    constexpr int N_LEVELS = 1 << BIT_WIDTH;
    constexpr int N_BOUNDARIES = N_LEVELS - 1;
    constexpr int PACKED_PER_GROUP = PackedSize<GROUP_DIM, BIT_WIDTH>::value;

    const int token = blockIdx.x;
    if (token >= T) return;
    const int tid = threadIdx.x;

    __shared__ float s_vec[GROUP_DIM];
    __shared__ float s_rotated[GROUP_DIM];
    __shared__ uint8_t s_indices[GROUP_DIM];
    __shared__ float s_partial[4];

    const __hip_bfloat16* kv_row = kv_data + (int64_t)token * (kv_lora_rank + qk_rope_head_dim);
    uint8_t* out_row = output + (int64_t)token * compressed_bytes;

    const int packed_total = (kv_lora_rank * BIT_WIDTH + 7) / 8;
    // Norms offset depends on bit_width packing
    int norms_offset;
    if constexpr (BIT_WIDTH == 4) norms_offset = kv_lora_rank / 2;
    else if constexpr (BIT_WIDTH == 3) norms_offset = kv_lora_rank * 3 / 8;
    else norms_offset = kv_lora_rank / 4;

    const float scale = sqrtf((float)GROUP_DIM);

    for (int g = 0; g < n_groups; g++) {
        const int g_start = g * GROUP_DIM;
        const float* Pi_g = Pi_all + (int64_t)g * GROUP_DIM * GROUP_DIM;

        // Step 1: Load + L2 norm
        float val = 0.0f;
        if (tid < GROUP_DIM && g_start + tid < kv_lora_rank) {
            val = __bfloat162float(kv_row[g_start + tid]);
        }
        float sq = val * val;

        for (int offset = 32; offset > 0; offset >>= 1)
            sq += __shfl_xor(sq, offset);

        int warp_id = tid / 64;
        int lane_id = tid % 64;
        if (lane_id == 0) s_partial[warp_id] = sq;
        __syncthreads();

        float norm_sq = 0.0f;
        if (tid < 4)
            norm_sq = (tid < (GROUP_DIM + 63) / 64) ? s_partial[tid] : 0.0f;
        if (tid < 4) {
            for (int offset = 2; offset > 0; offset >>= 1)
                norm_sq += __shfl_xor(norm_sq, offset);
            s_partial[tid] = norm_sq;
        }
        __syncthreads();
        norm_sq = s_partial[0];

        float norm = fmaxf(sqrtf(norm_sq), 1e-8f);

        if (tid == 0) {
            __half norm_h = __float2half(norm);
            uint16_t bits;
            memcpy(&bits, &norm_h, 2);
            out_row[norms_offset + g * 2]     = (uint8_t)(bits & 0xFF);
            out_row[norms_offset + g * 2 + 1] = (uint8_t)((bits >> 8) & 0xFF);
        }

        // Step 2: Normalize
        float normalized = val / norm;
        if (tid < GROUP_DIM) s_vec[tid] = normalized;
        __syncthreads();

        // Step 3: Rotate
        float y = 0.0f;
        if (tid < GROUP_DIM) {
            const float* Pi_row = Pi_g + tid * GROUP_DIM;
            for (int j = 0; j < GROUP_DIM; j++)
                y += Pi_row[j] * s_vec[j];
        }

        // Step 4: Scale + searchsorted
        y *= scale;
        int idx = 0;
        if (tid < GROUP_DIM) {
            for (int b = 0; b < N_BOUNDARIES; b++)
                if (y > boundaries[b]) idx = b + 1;
        }
        if (tid < GROUP_DIM) s_indices[tid] = (uint8_t)idx;
        __syncthreads();

        // Step 5: Pack indices
        if constexpr (BIT_WIDTH == 4) {
            if (tid < GROUP_DIM / 2) {
                uint8_t lo = s_indices[tid * 2];
                uint8_t hi = s_indices[tid * 2 + 1];
                out_row[g * PACKED_PER_GROUP + tid] = lo | (hi << 4);
            }
        } else if constexpr (BIT_WIDTH == 2) {
            if (tid < GROUP_DIM / 4) {
                uint8_t v0 = s_indices[tid * 4];
                uint8_t v1 = s_indices[tid * 4 + 1];
                uint8_t v2 = s_indices[tid * 4 + 2];
                uint8_t v3 = s_indices[tid * 4 + 3];
                out_row[g * PACKED_PER_GROUP + tid] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
            }
        } else if constexpr (BIT_WIDTH == 3) {
            // 8 values -> 3 bytes. Thread tid handles group-of-8 #(tid).
            if (tid < GROUP_DIM / 8) {
                int base = tid * 8;
                uint8_t v0 = s_indices[base], v1 = s_indices[base+1], v2 = s_indices[base+2];
                uint8_t v3 = s_indices[base+3], v4 = s_indices[base+4], v5 = s_indices[base+5];
                uint8_t v6 = s_indices[base+6], v7 = s_indices[base+7];
                int out_base = g * PACKED_PER_GROUP + tid * 3;
                out_row[out_base]     = v0 | (v1 << 3) | ((v2 & 0x03) << 6);
                out_row[out_base + 1] = (v2 >> 2) | (v3 << 1) | (v4 << 4) | ((v5 & 0x01) << 7);
                out_row[out_base + 2] = (v5 >> 1) | (v6 << 2) | (v7 << 5);
            }
        }
        __syncthreads();
    }

    // Copy rope (bf16 -> fp16 bytes)
    const __hip_bfloat16* rope_src = kv_row + kv_lora_rank;
    uint8_t* rope_dst = out_row + norms_offset + n_groups * 2;
    for (int i = tid; i < qk_rope_head_dim; i += blockDim.x) {
        float val = __bfloat162float(rope_src[i]);
        __half h = __float2half(val);
        uint16_t bits;
        memcpy(&bits, &h, 2);
        rope_dst[i * 2]     = (uint8_t)(bits & 0xFF);
        rope_dst[i * 2 + 1] = (uint8_t)((bits >> 8) & 0xFF);
    }
}


// ─── Decompress Kernel ───
template <int GROUP_DIM, int BIT_WIDTH>
__global__ void tq_kv_decompress_kernel(
    const uint8_t* __restrict__ input,
    const float* __restrict__ Pi_all,
    const float* __restrict__ centroids,
    __hip_bfloat16* __restrict__ kv_out,
    int T, int kv_lora_rank, int qk_rope_head_dim,
    int n_groups, int compressed_bytes)
{
    constexpr int PACKED_PER_GROUP = PackedSize<GROUP_DIM, BIT_WIDTH>::value;

    const int token = blockIdx.x;
    if (token >= T) return;
    const int tid = threadIdx.x;

    __shared__ float s_dequant[GROUP_DIM];

    const uint8_t* in_row = input + (int64_t)token * compressed_bytes;
    __hip_bfloat16* kv_row = kv_out + (int64_t)token * (kv_lora_rank + qk_rope_head_dim);

    int norms_offset;
    if constexpr (BIT_WIDTH == 4) norms_offset = kv_lora_rank / 2;
    else if constexpr (BIT_WIDTH == 3) norms_offset = kv_lora_rank * 3 / 8;
    else norms_offset = kv_lora_rank / 4;

    const float inv_scale = 1.0f / sqrtf((float)GROUP_DIM);

    for (int g = 0; g < n_groups; g++) {
        const int g_start = g * GROUP_DIM;
        const float* Pi_g = Pi_all + (int64_t)g * GROUP_DIM * GROUP_DIM;

        // Step 1: Unpack
        int idx = 0;
        if (tid < GROUP_DIM) {
            if constexpr (BIT_WIDTH == 4) {
                int byte_pos = g * PACKED_PER_GROUP + tid / 2;
                uint8_t packed_byte = in_row[byte_pos];
                idx = (tid % 2 == 0) ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
            } else if constexpr (BIT_WIDTH == 2) {
                int byte_pos = g * PACKED_PER_GROUP + tid / 4;
                uint8_t packed_byte = in_row[byte_pos];
                int shift = (tid % 4) * 2;
                idx = (packed_byte >> shift) & 0x03;
            } else if constexpr (BIT_WIDTH == 3) {
                int group8 = tid / 8;
                int pos_in_group = tid % 8;
                int byte_base = g * PACKED_PER_GROUP + group8 * 3;
                uint8_t b0 = in_row[byte_base], b1 = in_row[byte_base+1], b2 = in_row[byte_base+2];
                uint32_t bits24 = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16);
                idx = (bits24 >> (pos_in_group * 3)) & 0x07;
            }
        }

        // Step 2: Codebook + scale
        float dq = (tid < GROUP_DIM) ? centroids[idx] * inv_scale : 0.0f;
        s_dequant[tid] = dq;
        __syncthreads();

        // Step 3: Inverse rotate
        float x_val = 0.0f;
        if (tid < GROUP_DIM) {
            for (int j = 0; j < GROUP_DIM; j++)
                x_val += s_dequant[j] * Pi_g[j * GROUP_DIM + tid];
        }

        // Step 4: Denormalize
        uint16_t norm_bits = (uint16_t)in_row[norms_offset + g * 2]
                           | ((uint16_t)in_row[norms_offset + g * 2 + 1] << 8);
        __half norm_h;
        memcpy(&norm_h, &norm_bits, 2);
        x_val *= __half2float(norm_h);

        if (tid < GROUP_DIM && g_start + tid < kv_lora_rank)
            kv_row[g_start + tid] = __float2bfloat16(x_val);
        __syncthreads();
    }

    // Rope
    const uint8_t* rope_src = in_row + norms_offset + n_groups * 2;
    __hip_bfloat16* rope_dst = kv_row + kv_lora_rank;
    for (int i = tid; i < qk_rope_head_dim; i += blockDim.x) {
        uint16_t bits = (uint16_t)rope_src[i * 2] | ((uint16_t)rope_src[i * 2 + 1] << 8);
        __half h;
        memcpy(&h, &bits, 2);
        rope_dst[i] = __float2bfloat16(__half2float(h));
    }
}


// ─── Dispatch helpers ───

static int packed_bytes(int kv_lora_rank, int bit_width) {
    switch (bit_width) {
        case 4: return kv_lora_rank / 2;
        case 3: return kv_lora_rank * 3 / 8;
        case 2: return kv_lora_rank / 4;
        default: return kv_lora_rank / 2;
    }
}

#define DISPATCH_BIT_WIDTH(BW, BODY) \
    switch (BW) { \
        case 4: { constexpr int _BW = 4; BODY; break; } \
        case 3: { constexpr int _BW = 3; BODY; break; } \
        case 2: { constexpr int _BW = 2; BODY; break; } \
        default: TORCH_CHECK(false, "Unsupported bit_width: ", BW); \
    }


// ─── Python bindings ───

torch::Tensor turboquant_kv_compress(
    torch::Tensor kv_data, torch::Tensor Pi_all, torch::Tensor boundaries_t,
    int n_groups, int group_size, int bit_width)
{
    TORCH_CHECK(kv_data.dim() == 2 && kv_data.scalar_type() == torch::kBFloat16);
    const int T = kv_data.size(0);
    const int kv_cache_dim = kv_data.size(1);
    const int kv_lora_rank = n_groups * group_size;
    const int qk_rope_head_dim = kv_cache_dim - kv_lora_rank;

    const int pb = packed_bytes(kv_lora_rank, bit_width);
    const int compressed_bytes = pb + n_groups * 2 + qk_rope_head_dim * 2;

    auto output = torch::empty({T, compressed_bytes}, torch::dtype(torch::kUInt8).device(kv_data.device()));
    if (T == 0) return output;

    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(group_size == 128, "Only group_size=128 supported");

    DISPATCH_BIT_WIDTH(bit_width,
        tq_kv_compress_kernel<128, _BW><<<T, 128, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(kv_data.data_ptr()),
            Pi_all.data_ptr<float>(), boundaries_t.data_ptr<float>(),
            output.data_ptr<uint8_t>(),
            T, kv_lora_rank, qk_rope_head_dim, n_groups, compressed_bytes)
    );
    return output;
}

torch::Tensor turboquant_kv_decompress(
    torch::Tensor compressed, torch::Tensor Pi_all, torch::Tensor centroids_t,
    int n_groups, int group_size, int kv_lora_rank, int qk_rope_head_dim, int bit_width)
{
    TORCH_CHECK(compressed.dim() == 2 && compressed.scalar_type() == torch::kUInt8);
    const int T = compressed.size(0);
    const int compressed_bytes = compressed.size(1);
    const int kv_cache_dim = kv_lora_rank + qk_rope_head_dim;

    auto output = torch::empty({T, kv_cache_dim}, torch::dtype(torch::kBFloat16).device(compressed.device()));
    if (T == 0) return output;

    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(group_size == 128, "Only group_size=128 supported");

    DISPATCH_BIT_WIDTH(bit_width,
        tq_kv_decompress_kernel<128, _BW><<<T, 128, 0, stream>>>(
            compressed.data_ptr<uint8_t>(), Pi_all.data_ptr<float>(),
            centroids_t.data_ptr<float>(),
            reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
            T, kv_lora_rank, qk_rope_head_dim, n_groups, compressed_bytes)
    );
    return output;
}

// In-place variants for CUDA graph compatibility

void turboquant_kv_compress_inplace(
    torch::Tensor kv_data, torch::Tensor Pi_all, torch::Tensor boundaries_t,
    torch::Tensor output, int n_groups, int group_size, int bit_width)
{
    const int T = kv_data.size(0);
    const int kv_cache_dim = kv_data.size(1);
    const int kv_lora_rank = n_groups * group_size;
    const int qk_rope_head_dim = kv_cache_dim - kv_lora_rank;
    const int compressed_bytes = output.size(1);

    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_BIT_WIDTH(bit_width,
        tq_kv_compress_kernel<128, _BW><<<T, 128, 0, stream>>>(
            reinterpret_cast<const __hip_bfloat16*>(kv_data.data_ptr()),
            Pi_all.data_ptr<float>(), boundaries_t.data_ptr<float>(),
            output.data_ptr<uint8_t>(),
            T, kv_lora_rank, qk_rope_head_dim, n_groups, compressed_bytes)
    );
}

void turboquant_kv_decompress_inplace(
    torch::Tensor compressed, torch::Tensor Pi_all, torch::Tensor centroids_t,
    torch::Tensor output, int n_groups, int group_size,
    int kv_lora_rank, int qk_rope_head_dim, int bit_width)
{
    const int T = compressed.size(0);
    const int compressed_bytes = compressed.size(1);

    if (T == 0) return;
    auto stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_BIT_WIDTH(bit_width,
        tq_kv_decompress_kernel<128, _BW><<<T, 128, 0, stream>>>(
            compressed.data_ptr<uint8_t>(), Pi_all.data_ptr<float>(),
            centroids_t.data_ptr<float>(),
            reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
            T, kv_lora_rank, qk_rope_head_dim, n_groups, compressed_bytes)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turboquant_kv_compress", &turboquant_kv_compress);
    m.def("turboquant_kv_decompress", &turboquant_kv_decompress);
    m.def("turboquant_kv_compress_inplace", &turboquant_kv_compress_inplace);
    m.def("turboquant_kv_decompress_inplace", &turboquant_kv_decompress_inplace);
}
