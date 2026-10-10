// aiter entry for our 4-wave MXFP4 GEMM: reads aiter's asm_gemm_a4w4 KernelArgs, built into a .co for hsa/gfx950/f4gemm.
#define MXG4W_NO_ENTRY
#include "mxfp4_gemm_4w.hip"

struct p2 { unsigned int _p0, _p1; };
struct p3 { unsigned int _p0, _p1, _p2; };
struct KernelArgs {   // csrc/py_itfs_cu/asm_gemm_a4w4.cu, field for field (natural alignment = its packed layout)
    void* ptr_D; p2 _p0; void* ptr_C; p2 _p1; void* ptr_A; p2 _p2; void* ptr_B; p2 _p3;
    float alpha; p3 _p4; float beta; p3 _p5;
    unsigned int stride_D0; p3 _p6; unsigned int stride_D1; p3 _p7; unsigned int stride_C0; p3 _p8; unsigned int stride_C1; p3 _p9;
    unsigned int stride_A0; p3 _p10; unsigned int stride_A1; p3 _p11; unsigned int stride_B0; p3 _p12; unsigned int stride_B1; p3 _p13;
    unsigned int M; p3 _p14; unsigned int N; p3 _p15; unsigned int K; p3 _p16;
    void* ptr_ScaleA; p2 _p17; void* ptr_ScaleB; p2 _p18;
    unsigned int stride_ScaleA0; p3 _p19; unsigned int stride_ScaleA1; p3 _p20;
    unsigned int stride_ScaleB0; p3 _p21; unsigned int stride_ScaleB1; p3 _p22;
    int log2_k_split;
    unsigned int num_cu;
};
static_assert(__builtin_offsetof(KernelArgs, log2_k_split) == 368 && sizeof(KernelArgs) == 376);

__global__ __launch_bounds__(mxg4w::Threads, 1) void f4gemm_persistent(KernelArgs p)
    __asm__("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E");
__global__ void f4gemm_persistent(KernelArgs p)
{
#if defined(__HIP_DEVICE_COMPILE__)
    __shared__ __attribute__((aligned(1024))) uint8_t lds[mxg4w::LdsBytes];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / mxg4w::WaveSize), lane = threadIdx.x % mxg4w::WaveSize;
    asm volatile("" ::: "a255");                                         // the accumulators: all 256 AGPRs
    const int m = p.M, n = p.N, k = p.K;
    const mxg4w::Args args{(const uint8_t*)p.ptr_A, (const uint8_t*)p.ptr_B, (const uint8_t*)p.ptr_ScaleA,
                           (const uint8_t*)p.ptr_ScaleB, (__bf16*)p.ptr_D, m, n, k, (int)p.num_cu,
                           (int)p.stride_A0 / 2, (int)p.stride_ScaleA0, (int)p.stride_ScaleB0, (int)p.stride_C0, p.alpha, p.log2_k_split};
    mxg4w::run(args, lds, wave, lane);
#endif
}
