#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void spin(long long* out, long long iters) {
  long long c0 = clock64(), w0 = wall_clock64();
  float x = threadIdx.x;
  for (long long i = 0; i < iters; ++i) x = x * 1.000001f + 0.5f;
  long long c1 = clock64(), w1 = wall_clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = c1 - c0; out[1] = w1 - w0; out[2] = (long long)x; }
}
int main() {
  int wall_khz = 0, clk_khz = 0, cus = 0;
  hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0);
  hipDeviceGetAttribute(&clk_khz, hipDeviceAttributeClockRate, 0);
  hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0);
  long long* d; hipMalloc(&d, 3 * sizeof(long long));
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
  struct { int grid, block; long long iters; } runs[] = {{1, 32, 200000}, {1, 32, 2000000}, {cus * 4, 128, 2000000}};
  for (auto r : runs) {
    spin<<<r.grid, r.block>>>(d, 1000); hipDeviceSynchronize();
    hipEventRecord(a); spin<<<r.grid, r.block>>>(d, r.iters); hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    long long h[3]; hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost);
    double wall_s = (double)h[1] / (wall_khz * 1e3);
    printf("RESULT grid=%d block=%d iters=%lld cycles=%lld wall_ticks=%lld wall_khz=%d attr_clk_khz=%d event_ms=%.3f shader_MHz_by_wallclock=%.1f\n",
           r.grid, r.block, r.iters, h[0], h[1], wall_khz, clk_khz, ms, h[0] / wall_s / 1e6);
  }
  return 0;
}
