#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "hip_compat.h"
#include "dispatch_utils.h"
#include <torch/torch.h>
#include <cmath>

#ifdef USE_ROCM
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 nv_bfloat16;
#else
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

namespace aiter
{
    template <typename T, typename Operation>
    inline __device__ T performUnaryOperation(T a);

    struct TanhOp
    {
        template <typename T>
        inline __device__ static T apply(T a)
        {
            return (T)(::tanhf(static_cast<float>(a)));

            // float y, x = static_cast<float>(a);
            // float neg_x = -x;
            // const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
            // float tmp = 0, neg_tmp = 0, m = 0, n = 0, emu = 0, neg_emu = 0;
            // asm volatile(
            //              "v_mul_f32 %[v_neg_tmp], %[s_log2e], %[v_neg_x]; log2e*(-x)\n"
            //              "s_nop 8                                       ; hazard for exp\n"
            //              "v_mul_f32 %[v_tmp], %[s_log2e], %[v_x]        ; log2e*x\n"
            //              "s_nop 8                                       ; hazard for exp\n"
            //              "v_exp_f32 %[v_neg_emu], %[v_neg_tmp]          ; neg_emu = exp2(log2e*(-x)) 0.3678794515979072\n"
            //              "s_nop 8                                       ; hazard for exp\n"
            //              "v_exp_f32 %[v_emu], %[v_tmp]                  ; emu = exp2(log2e*x)\n"
            //              "s_nop 8                                       ; hazard for exp\n"
            //              "v_add_f32 %[v_m], %[v_emu], %[v_neg_emu]      ;m=emu+neg_emu\n"
            //              "v_sub_f32 %[v_n], %[v_emu], %[v_neg_emu]      ;n=emu - neg_emu\n"
            //              "v_rcp_f32 %[v_tmp], %[v_m]                      ; 1/m\n"
            //              "s_nop 4                                       ; hazard for rcp \n"
            //              "v_mul_f32 %[v_y], %[v_n], %[v_tmp]              ; n/m\n"
            //              "s_nop 8                                       ; hazard for exp\n"
            //              : [v_y] "=v"(y),
            //                [v_tmp] "+v"(tmp),
            //                [v_neg_tmp] "+v"(neg_tmp),
            //                [v_emu] "+v"(emu),
            //                [v_neg_emu] "+v"(neg_emu),
            //                [v_m] "+v"(m),
            //                [v_n] "+v"(n)
            //              : [v_x] "v"(x), [v_neg_x] "v"(neg_x), [s_log2e] "n" (log2e_)
            //              :);
            // return static_cast<T>(y);
        }

        static torch::Tensor compute(torch::Tensor &input)
        {
            return torch::tanh(input);
        }
    };

    struct SigmoidOp
    {
        template <typename T>
        inline __device__ static T apply(T x)
        {
            //   float y, neg_a = static_cast<float>(-x);
            //   const uint32_t log2e_ = 0x3fb8aa3b; // log2e_v<float>;
            //   float tmp;
            //   asm volatile("v_mul_f32 %[v_tmp], %[s_log2e], %[v_x]    ; log2e*x\n"
            //                "v_exp_f32 %[v_tmp], %[v_tmp]              ; emu = exp2(log2e*x)\n"
            //                "s_nop 4                                   ; hazard for exp\n"
            //                "v_add_f32 %[v_tmp], %[v_tmp], 1.0         ; emu+1.0f\n"
            //                "v_rcp_f32 %[v_y], %[v_tmp]                ; 1/(emu+1.0f)\n"
            //                "s_nop 4                                   ; hazard for rcp \n"
            //                : [v_y] "=v"(y), [v_tmp] "+v"(tmp)
            //                : [v_x] "v"(neg_a), [s_log2e] "n"(log2e_)
            //                :);
            //   return static_cast<T>(y);
            return static_cast<T>(1.0f / (1.0f + expf(static_cast<float>(-x))));
        }

        static torch::Tensor compute(torch::Tensor &input)
        {
            return torch::sigmoid(input);
        }
    };

    template <class _T, int _rows, int _vec, typename Operation>
    __global__ void unary_operator_tile_kernel(const void *__restrict a, void *__restrict c, const int M, const int N, const int K)
    {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t n_tiles = N / _rows;
        uint32_t k_tiles = K / _vec;
        if (idx < (uint64_t)M * n_tiles * k_tiles)
        {
            uint32_t ti = idx / (k_tiles * n_tiles);
            uint64_t idx_block = idx % (k_tiles * n_tiles);
            uint32_t tj = (idx_block / k_tiles) % n_tiles;
            uint32_t tk = idx_block % k_tiles;
            for (int row = 0; row < _rows; row++)
            {
                uint64_t offset_ac = (uint64_t)(tj + row * n_tiles) * K + tk * _vec + (uint64_t)ti * N * K;
                const _T *pa = (const _T *)a + offset_ac;
                _T *pc = (_T *)c + offset_ac;
                for (int col = 0; col < _vec; col++)
                {
                    const _T *pfa = (const _T *)(pa + col);
                    _T *pfc = (_T *)(pc + col);
                    *pfc = Operation::apply(*pfa);
                }
            }
        }
    }
}

template <typename Operation>
torch::Tensor unary_operation(torch::Tensor &input)
{
    int dim = input.dim();
    bool is_support = true;
    is_support &= input.is_contiguous() == true;
    int M = dim == 2 ? 1 : input.size(0);
    int N = dim == 2 ? input.size(0) : input.size(1);
    int K = dim == 2 ? input.size(1) : input.size(2);
    const uint32_t rows = 8;
    const uint32_t vec = 16 / sizeof(input.dtype());
    is_support &= N % rows == 0;
    is_support &= K % vec == 0;
    if (is_support)
    {
        auto options = torch::TensorOptions().dtype(input.dtype()).device("cuda");
        auto output = torch::empty(input.sizes(), options);
        void *buf_c = reinterpret_cast<void *>(output.data_ptr());

        void *buf_a = reinterpret_cast<void *>(input.data_ptr());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        int elements = N * K;

        constexpr uint32_t wg = 256;
        int grid_x = (elements / (rows * vec) + wg - 1) / wg;
        const dim3 grid_dim(grid_x, 1, 1);
        const dim3 block_dim(wg, 1, 1);

        VLLM_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "unary_operator_tile_kernel", [&]
            { aiter::unary_operator_tile_kernel<scalar_t, rows, vec, Operation>
                  <<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_c, M, N, K); });
        return output;
    }
    else
    {
        return Operation::compute(input);
    }
}

torch::Tensor aiter_sigmoid(torch::Tensor &input)
{
    return unary_operation<aiter::SigmoidOp>(input);
}

torch::Tensor aiter_tanh(torch::Tensor &input)
{
    return unary_operation<aiter::TanhOp>(input);
}
