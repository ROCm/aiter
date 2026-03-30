#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                 \
    do {                                                                \
        hipError_t err = (call);                                        \
        if (err != hipSuccess) {                                        \
            fprintf(stderr, "HIP error %s at %s:%d\n",                  \
                    hipGetErrorString(err), __FILE__, __LINE__);        \
            exit(1);                                                    \
        }                                                               \
    } while (0)

// Empty kernel with same arg count as moe_gemm2_0 tensor_api:
// 9 pointers + 2 ints = 11 args
__global__ void empty_kernel_11args(
    void* __restrict__ out,
    const void* __restrict__ inter_states,
    const void* __restrict__ w2,
    const void* __restrict__ a_scale,
    const void* __restrict__ w_scale,
    const void* __restrict__ sorted_ids,
    const void* __restrict__ sorted_expert_ids,
    const void* __restrict__ sorted_weights,
    const void* __restrict__ num_valid_ids,
    int token_num,
    int valid_m_blocks)
{
    // completely empty
}

// For comparison: minimal 1-arg kernel
__global__ void empty_kernel_1arg(void* __restrict__ out)
{
}

// For comparison: kernel with 20 args (simulating heavier arg passing)
__global__ void empty_kernel_20args(
    void* a0, void* a1, void* a2, void* a3, void* a4,
    void* a5, void* a6, void* a7, void* a8, void* a9,
    void* a10, void* a11, void* a12, void* a13, void* a14,
    int i0, int i1, int i2, int i3, int i4)
{
}

double bench_kernel(void (*launch_fn)(hipStream_t), int warmup, int iters,
                    hipStream_t stream)
{
    for (int i = 0; i < warmup; i++)
        launch_fn(stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < iters; i++)
        launch_fn(stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms * 1000.0 / iters;  // us per launch
}

static void* d_buf = nullptr;

void launch_1arg(hipStream_t s)  { empty_kernel_1arg<<<1,1,0,s>>>(d_buf); }
void launch_11args(hipStream_t s) {
    empty_kernel_11args<<<1,256,0,s>>>(
        d_buf, d_buf, d_buf, d_buf, d_buf,
        d_buf, d_buf, d_buf, d_buf, 1, 1);
}
void launch_11args_grid(hipStream_t s) {
    // grid similar to stage2: gx=56 (7168/128), gy=8 (topk expert blocks)
    empty_kernel_11args<<<dim3(56,8,1),256,0,s>>>(
        d_buf, d_buf, d_buf, d_buf, d_buf,
        d_buf, d_buf, d_buf, d_buf, 1, 8);
}
void launch_20args(hipStream_t s) {
    empty_kernel_20args<<<1,256,0,s>>>(
        d_buf, d_buf, d_buf, d_buf, d_buf,
        d_buf, d_buf, d_buf, d_buf, d_buf,
        d_buf, d_buf, d_buf, d_buf, d_buf,
        0, 0, 0, 0, 0);
}

int main()
{
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_buf, 4096));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int warmup = 200, iters = 1000;
    printf("Measuring empty kernel launch overhead (warmup=%d, iters=%d)\n\n",
           warmup, iters);

    double us1  = bench_kernel(launch_1arg,       warmup, iters, stream);
    double us11 = bench_kernel(launch_11args,      warmup, iters, stream);
    double us11g= bench_kernel(launch_11args_grid, warmup, iters, stream);
    double us20 = bench_kernel(launch_20args,      warmup, iters, stream);

    printf("  empty_kernel_1arg   (grid=1x1,  block=1):    %.2f us\n", us1);
    printf("  empty_kernel_11args (grid=1x1,  block=256):  %.2f us\n", us11);
    printf("  empty_kernel_11args (grid=56x8, block=256):  %.2f us\n", us11g);
    printf("  empty_kernel_20args (grid=1x1,  block=256):  %.2f us\n", us20);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_buf));
    return 0;
}
