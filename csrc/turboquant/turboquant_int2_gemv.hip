// TurboQuant INT2 Weight GEMV: bf16 x (M,K) × int2 W (N,K) → bf16 y (M,N)
//
// Optimized for decode (M=1-8). Uses LDS for x vector and codebook,
// loads INT2 weights directly from global memory with per-group norm scaling.
//
// Layout: W is stored as (N, K/4) uint8, each byte holds 4 2-bit indices.
// No preshuffle needed — straightforward packed row-major.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// 2-bit Lloyd-Max centroids for N(0,1)
__device__ __constant__ float TQ2_CENTROIDS[4] = {
    -1.5104176998f, -0.4527800398f, 0.4527800398f, 1.5104176998f
};

// ─── INT2 GEMV kernel (warp-cooperative, coalesced) ───
// Each warp handles one N-row. 64 threads reduce over K dimension.
// Adjacent threads load adjacent bytes from the same row → coalesced.
// Grid: (N, M)   Block: (64,)  — one warp per block per N-row
template <int GROUP_SIZE>
__global__ void tq_int2_gemv_kernel(
    const __hip_bfloat16* __restrict__ x,    // (M, K)
    const uint8_t* __restrict__ w_packed,     // (N, K/4)
    const float* __restrict__ norms,          // (N, n_groups)
    __hip_bfloat16* __restrict__ y,           // (M, N)
    int M, int N, int K,
    int norms_stride,
    float inv_sqrt_gs)
{
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int tid = threadIdx.x;
    const int K_bytes = K / 4;

    const __hip_bfloat16* x_row = x + m * K;
    const uint8_t* w_row = w_packed + (int64_t)n * K_bytes;

    const int n_groups = K / GROUP_SIZE;
    float acc = 0.0f;

    for (int g = 0; g < n_groups; g++) {
        const int g_byte_start = g * (GROUP_SIZE / 4);
        const int bytes_per_group = GROUP_SIZE / 4;  // 32 bytes for gs=128
        float group_acc = 0.0f;

        // Each of 64 threads handles bytes_per_group/64 bytes (or partial)
        for (int b = tid; b < bytes_per_group; b += 64) {
            uint8_t packed = w_row[g_byte_start + b];
            int k_base = g * GROUP_SIZE + b * 4;

            float x0 = __bfloat162float(x_row[k_base]);
            float x1 = __bfloat162float(x_row[k_base + 1]);
            float x2 = __bfloat162float(x_row[k_base + 2]);
            float x3 = __bfloat162float(x_row[k_base + 3]);

            group_acc += x0 * TQ2_CENTROIDS[(packed >> 0) & 3]
                       + x1 * TQ2_CENTROIDS[(packed >> 2) & 3]
                       + x2 * TQ2_CENTROIDS[(packed >> 4) & 3]
                       + x3 * TQ2_CENTROIDS[(packed >> 6) & 3];
        }

        // Warp reduction (64 threads → 1 value)
        for (int offset = 32; offset > 0; offset >>= 1)
            group_acc += __shfl_xor(group_acc, offset);

        if (tid == 0) {
            float ns = norms[n * norms_stride + g] * inv_sqrt_gs;
            acc += group_acc * ns;
        }
    }

    if (tid == 0) {
        y[m * N + n] = __float2bfloat16(acc);
    }
}

// ─── Batched version: multiple N-rows per block for better occupancy ───
template <int GROUP_SIZE, int ROWS_PER_BLOCK>
__global__ void tq_int2_gemv_batched_kernel(
    const __hip_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const float* __restrict__ norms,
    __hip_bfloat16* __restrict__ y,
    int M, int N, int K,
    int norms_stride,
    float inv_sqrt_gs)
{
    // Each block handles ROWS_PER_BLOCK N-rows
    // Block: (64 * ROWS_PER_BLOCK,) — 64 threads per row
    const int row_in_block = threadIdx.x / 64;
    const int tid = threadIdx.x % 64;
    const int n = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;

    __shared__ float s_x[512];  // max K=512
    // Cooperative load x into shared memory
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        s_x[k] = __bfloat162float(x[m * K + k]);
    __syncthreads();

    const int K_bytes = K / 4;
    const uint8_t* w_row = w_packed + (int64_t)n * K_bytes;
    const int n_groups = K / GROUP_SIZE;
    float acc = 0.0f;

    for (int g = 0; g < n_groups; g++) {
        const int g_byte_start = g * (GROUP_SIZE / 4);
        const int bytes_per_group = GROUP_SIZE / 4;
        float group_acc = 0.0f;

        for (int b = tid; b < bytes_per_group; b += 64) {
            uint8_t packed = w_row[g_byte_start + b];
            int k_base = g * GROUP_SIZE + b * 4;

            group_acc += s_x[k_base]     * TQ2_CENTROIDS[(packed >> 0) & 3]
                       + s_x[k_base + 1] * TQ2_CENTROIDS[(packed >> 2) & 3]
                       + s_x[k_base + 2] * TQ2_CENTROIDS[(packed >> 4) & 3]
                       + s_x[k_base + 3] * TQ2_CENTROIDS[(packed >> 6) & 3];
        }

        for (int offset = 32; offset > 0; offset >>= 1)
            group_acc += __shfl_xor(group_acc, offset);

        if (tid == 0) {
            float ns = norms[n * norms_stride + g] * inv_sqrt_gs;
            acc += group_acc * ns;
        }
    }

    if (tid == 0) {
        y[m * N + n] = __float2bfloat16(acc);
    }
}


// ─── Python binding ───

torch::Tensor turboquant_int2_gemv(
    torch::Tensor x_rot,         // (M, K) bf16 — pre-rotated input
    torch::Tensor w_packed,      // (N, K/4) uint8 — packed INT2 weights
    torch::Tensor norms,         // (N, n_groups) float32
    int group_size)
{
    TORCH_CHECK(x_rot.dim() == 2 && x_rot.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w_packed.dim() == 2 && w_packed.scalar_type() == torch::kUInt8);

    const int M = x_rot.size(0);
    const int K = x_rot.size(1);
    const int N = w_packed.size(0);

    TORCH_CHECK(w_packed.size(1) == K / 4, "w_packed must be (N, K/4)");
    TORCH_CHECK(group_size == 128, "Only group_size=128 supported");

    auto output = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_rot.device()));

    const int n_groups = K / group_size;
    const int norms_stride = (norms.dim() == 2) ? n_groups : 0;
    const float inv_sqrt_gs = 1.0f / sqrtf(static_cast<float>(group_size));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Use batched kernel: 4 N-rows per block, 64 threads per row = 256 threads/block
    constexpr int ROWS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = 64 * ROWS_PER_BLOCK;
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, M);
    tq_int2_gemv_batched_kernel<128, ROWS_PER_BLOCK><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(x_rot.data_ptr()),
        w_packed.data_ptr<uint8_t>(),
        norms.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        M, N, K, norms_stride, inv_sqrt_gs
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turboquant_int2_gemv", &turboquant_int2_gemv,
          "TurboQuant INT2 weight GEMV with per-group norms");
}
