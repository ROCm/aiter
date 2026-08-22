#include <hip/hip_runtime.h>
#include <cstdio>

using f32x16 = float __attribute__((ext_vector_type(16)));

// 4 independent accumulator chains per wave so the matrix pipe is throughput-bound,
// not latency-bound. Each wave issues ONLY v_mfma_f32_32x32x16_fp8_fp8.
__global__ __launch_bounds__(256) void mfma_peak(float* out, long a, long b, int iters) {
    f32x16 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, c3, 0, 0, 0);
    }
    f32x16 s = c0 + c1 + c2 + c3;
    float acc = 0;
    for (int k = 0; k < 16; ++k) acc += s[k];
    if (acc == -1.0f) out[threadIdx.x] = acc;   // never true: keeps MFMAs live, no store traffic
}

#define CK(x) do{ hipError_t e=(x); if(e){printf("HIP err %s @%d: %s\n",#x,__LINE__,hipGetErrorString(e));return 1;} }while(0)

int main() {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
    int cu = p.multiProcessorCount;
    double clkGHz = p.clockRate / 1e6;   // clockRate is kHz (max boost)
    printf("Device: %s | CUs=%d | maxClock=%.3f GHz\n\n", p.gcnArchName, cu, clkGHz);

    const int MFMA_PER_ITER = 4;
    const long MMAC_FLOP = 2LL*32*32*16;         // 32768 flop per 32x32x16 mfma
    int iters = 200000;
    float* out=nullptr; CK(hipMalloc(&out, 1024*sizeof(float)));
    long a = 0x3f3f3f3f3f3f3f3fLL, b = 0x3c3c3c3c3c3c3c3cLL;

    hipEvent_t evs,eve; CK(hipEventCreate(&evs)); CK(hipEventCreate(&eve));
    printf("%-14s %-10s %8s %10s %8s\n","waves/CU(occ)","total_wv","time_ms","TFLOPS","cyc/mfma@max");
    int wpc_list[] = {1,2,4,8,16};   // waves per CU
    for (int wpc : wpc_list) {
        int block = 256;                         // 4 waves per workgroup
        int wg_per_cu = (wpc + 3)/4;             // workgroups per CU to reach wpc waves
        if (wpc < 4) { block = wpc*64; wg_per_cu = 1; }
        int grid = cu * wg_per_cu;
        int real_wpc = (block/64)*wg_per_cu;
        mfma_peak<<<grid, block>>>(out, a, b, 1000); CK(hipDeviceSynchronize()); // warmup
        CK(hipEventRecord(evs));
        mfma_peak<<<grid, block>>>(out, a, b, iters);
        CK(hipEventRecord(eve)); CK(hipEventSynchronize(eve));
        float ms=0; CK(hipEventElapsedTime(&ms, evs, eve));
        long waves = (long)grid * (block/64);
        double total_mfma = (double)waves * iters * MFMA_PER_ITER;
        double tflops = total_mfma * MMAC_FLOP / (ms/1e3) / 1e12;
        double flop_per_simd_cyc = tflops*1e12 / (cu*4.0 * clkGHz*1e9);
        printf("%-14d %-10ld %8.1f %10.1f %8.1f\n", real_wpc, waves, ms, tflops, 32768.0/flop_per_simd_cyc);
    }
    return 0;
}
