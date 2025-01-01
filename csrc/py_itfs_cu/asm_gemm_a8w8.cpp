#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "ater_hip_common.h"

// start to prepare the input and output buffer
struct __attribute__((packed)) KernelArgs
{
    void *ptr_c;
    p2 _p0;
    void *ptr_a;
    p2 _p1;
    void *ptr_b;
    p2 _p2;
    void *ptr_sa;
    p2 _p3;
    void *ptr_sb;
    p2 _p4;
    void *ptr_bias;
    p2 _p5;
    // float alpha;
    unsigned int m;
    p3 _p12;
    unsigned int n;
    p3 _p13;
    unsigned int k;
    p3 _p14;
    unsigned int lda;
    p3 _p15;
    unsigned int ldb;
    p3 _p16;
    unsigned int ldc;
    p3 _p17;
    unsigned int ks;
    p3 _p18;
};

class GemmA8W8Kernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    GemmA8W8Kernel(const char *name, const char *hsaco)
    {
        HIP_CALL(hipModuleLoad(&module, (std::string(ATER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    };

    template <typename T_O>
    void launch_kernel(torch::Tensor &out,     // Out:[M, N] bf16
                       torch::Tensor &A,       // A:[M, K] i8
                       torch::Tensor &B,       //  B:[N, K] i8 -> shuffle layout(32,16)
                       torch::Tensor &A_scale, // A_scale:[M, 1] f32
                       torch::Tensor &B_scale, // B_scale:[1, N] f32
                       torch::Tensor &bias,    // bias:[1, N] f32
                       int sub_m = 128,
                       int sub_n = 128,
                       int pad_a = 0,
                       int pad_b = 0,
                       int pad_c = 0,
                       int splitK = 0)
    {
        int m = A.size(0);
        int n = out.size(1);
        int k = A.size(1);
        int stride_a = k + pad_a;
        int stride_b = k + pad_b;
        int stride_c = n + pad_c;
        stride_c = stride_c * sizeof(T_O);

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_c = (void *)out.data_ptr();
        args.ptr_a = (void *)A.data_ptr();
        args.ptr_b = (void *)B.data_ptr();
        args.ptr_sa = (void *)A_scale.data_ptr();
        args.ptr_sb = (void *)B_scale.data_ptr();
        args.ptr_bias = (void *)bias.data_ptr();
        // args.alpha  = alpha;
        args.m = m;
        args.n = n;
        args.k = k;
        args.lda = stride_a;
        args.ldb = stride_b;
        args.ldc = stride_c;
        args.ks = splitK;

        // std::cout << "m:"            << m          << std::endl;
        // std::cout << "n:"            << n          << std::endl;
        // std::cout << "k:"            << k          << std::endl;
        // std::cout << "pad_a:"        << pad_a      << std::endl;
        // std::cout << "pad_b:"        << pad_b      << std::endl;
        // std::cout << "pad_c:"        << pad_c      << std::endl;
        // std::cout << "sub_m:"        << sub_m      << std::endl;
        // std::cout << "sub_n:"        << sub_n      << std::endl;
        // std::cout << "ks:"           << args.ks    << std::endl;
        // std::cout << "lda:"          << args.lda   << std::endl;
        // std::cout << "ldb:"          << args.ldb   << std::endl;
        // std::cout << "ldc:"          << args.ldc   << std::endl;

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int bdy = 1;
        int bdz = 1;
        int gdx = n / sub_n;
        int gdy = m / sub_m;
        int gdz = 1;
        gdy = gdy << splitK; // shift for ksplit
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx, gdy, gdz,
                                       bdx, bdy, bdz,
                                       0, stream, nullptr, (void **)&config));
    };
};

torch::Tensor gemm_a8w8_asm(torch::Tensor &A,       // A:[M, K] i8
                            torch::Tensor &B,       // B:[N, K] i8 -> shuffle layout(32,16)
                            torch::Tensor &A_scale, // A_scale:[M, 1] f32
                            torch::Tensor &B_scale, // B_scale:[1, N] f32
                            torch::Tensor &out,     // Out:[M, N] bf16
                            torch::Tensor &bias,    // bias:[1, N] f32
                            std::optional<int> sub_m = 128,
                            std::optional<int> sub_n = 128,
                            std::optional<int> pad_a = 0,
                            std::optional<int> pad_b = 0,
                            std::optional<int> pad_c = 0,
                            std::optional<int> splitK = 0)
{
    if (splitK > 0)
    {
        static GemmA8W8Kernel splitK_impl("gemm_kernel_func", "gemm_a8w8_m128_splitK.co");
        splitK_impl.launch_kernel<uint16_t>(out,
                                            A,
                                            B,
                                            A_scale,
                                            B_scale,
                                            bias,
                                            sub_m.value(),
                                            sub_n.value(),
                                            pad_a.value(),
                                            pad_b.value(),
                                            pad_c.value(),
                                            splitK.value());
    }
    else
    {
        static GemmA8W8Kernel noSplitK_impl("gemm_kernel_func", "gemm_a8w8_m128_noSplitK.co");
        noSplitK_impl.launch_kernel<uint16_t>(out,
                                              A,
                                              B,
                                              A_scale,
                                              B_scale,
                                              bias,
                                              sub_m.value(),
                                              sub_n.value(),
                                              pad_a.value(),
                                              pad_b.value(),
                                              pad_c.value());
    }
    return out;
}
