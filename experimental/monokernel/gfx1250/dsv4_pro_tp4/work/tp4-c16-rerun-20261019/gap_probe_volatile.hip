// Host time-slicing probe: one spinning wave per CTA records the largest gap between consecutive s_memrealtime
// reads (100 MHz). Build: hipcc -O2 -o gap gap_probe.hip   Run: ./gap <ctas> <seconds>
// Run alone, then in 2 and 8 concurrent processes (one per GPU, HIP_VISIBLE_DEVICES=i). Healthy: max gap < 1 us,
// no gaps > 1 ms. ~100 ms gaps mean the host deschedules GPU processes and multi-process timings are invalid.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__device__ __forceinline__ long long read_timer() {
    long long value;
    asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n s_wait_kmcnt 0" : "=s"(value) :: "memory");
    return value;
}
__global__ void gap(long long* out, long long ticks) {
    long long t0 = read_timer();
    long long prev = t0, maxgap = 0, ngap = 0, now = t0;
    while (now - t0 < ticks) {
        now = read_timer();
        long long g = now - prev;
        if (g > maxgap) maxgap = g;
        if (g > 100000) ngap++;
        prev = now;
    }
    if (threadIdx.x == 0) { out[blockIdx.x * 3] = maxgap; out[blockIdx.x * 3 + 1] = ngap; out[blockIdx.x * 3 + 2] = now - t0; }
}
int main(int argc, char** argv) {
    int nb = argc > 1 ? atoi(argv[1]) : 1;
    double secs = argc > 2 ? atof(argv[2]) : 1.0;
    long long* d; hipMalloc(&d, nb * 3 * sizeof(long long));
    long long* h = (long long*)malloc(nb * 3 * sizeof(long long));
    for (int rep = 0; rep < 2; rep++) {
        gap<<<nb, 64>>>(d, (long long)(secs * 1e8));
        hipDeviceSynchronize();
        hipMemcpy(h, d, nb * 3 * sizeof(long long), hipMemcpyDeviceToHost);
        long long mg = 0, ng = 0; for (int i = 0; i < nb; i++) { if (h[i*3] > mg) mg = h[i*3]; if (h[i*3+1] > ng) ng = h[i*3+1]; }
        printf("rep %d grid %d: max gap %.1f us, gaps>1ms (max per CTA) %lld, span %.1f ms\n", rep, nb, mg / 100.0, ng, h[2] / 100.0 / 1e3);
    }
    return 0;
}
