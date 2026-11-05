// IOD phase probe for aiter/ops/flydsl/iod_locality.py (MI455X SPX/NPS1, gfx1250).
// home(addr) = a14 ^ a17 ^ a20 ^ phase[addr >> 21];  WG domain = ((blockIdx + 3) % 8) >= 4.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>

__device__ __forceinline__ uint32_t timed_load(const uint32_t* p, uint32_t& v) {
  uint64_t t0, t1;
  asm volatile(
      "s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\t"
      "s_wait_kmcnt 0\n\t"
      "global_load_b32 %2, %3, off\n\t"
      "s_wait_loadcnt 0\n\t"
      "s_sendmsg_rtn_b64 %1, sendmsg(MSG_RTN_GET_REALTIME)\n\t"
      "s_wait_kmcnt 0"
      : "=&s"(t0), "=&s"(t1), "=&v"(v)
      : "v"(p)
      : "memory");
  return (uint32_t)(t1 - t0);
}

__device__ void grid_sync(uint32_t* bar, uint32_t nwg, uint32_t& epoch) {
  epoch++;
  __hip_atomic_fetch_add(bar, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  while (__hip_atomic_load(bar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < epoch * nwg) {}
}

__host__ __device__ inline int wg_dom(uint32_t b) { return ((b + 3) & 7) >= 4; }
__host__ __device__ inline int inpage(uint64_t off) { return ((off >> 14) ^ (off >> 17) ^ (off >> 20)) & 1; }
__host__ __device__ inline uint64_t probe_off(uint32_t b, uint32_t kline, uint32_t cls, uint32_t pass) {
  return ((uint64_t)b << 21) + kline * 65536ull + pass * 4096ull + 128 * cls;
}

__global__ void probe_k(const char* buf, uint16_t* lat, uint32_t* bar, uint32_t npg, uint32_t pass) {
  extern __shared__ uint32_t lds[];
  if (threadIdx.x != 0) return;
  lds[0] = 0;
  const uint32_t nwg = gridDim.x, j = blockIdx.x, kline = j / 8, cls = j % 8;
  uint32_t epoch = 0, v;
  grid_sync(bar, nwg, epoch);
  for (uint32_t p = 0; p < nwg; p++) {
    if (p == j)
      for (uint32_t b = 0; b < npg; b++) {
        if ((((b * 2654435761u) >> 29) + kline + pass) % 8 != cls) continue;
        uint32_t t = timed_load((const uint32_t*)(buf + probe_off(b, kline, cls, pass)), v);
        lat[b * 32 + kline] = (uint16_t)(t > 65535 ? 65535 : t);
      }
    grid_sync(bar, nwg, epoch);
  }
}

// phase[npg] out (host); returns the number of pages whose vote margin is below 75 %.
extern "C" int loc_probe(void* buf, uint32_t npg, uint8_t* phase, int npass) {
  const uint32_t nwg = 256;
  uint16_t* lat;
  uint32_t* bar;
  void* flush;
  const size_t nflush = 1ull << 30;
  if (hipMalloc(&lat, (size_t)npg * 32 * 2) || hipMalloc(&bar, 4) || hipMalloc(&flush, nflush)) return -1;
  (void)hipFuncSetAttribute((const void*)probe_k, hipFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  std::vector<int> votes(npg, 0), n(npg, 0);
  std::vector<uint16_t> h((size_t)npg * 32);
  for (int pass = 0; pass < npass; pass++) {
    (void)hipMemset(flush, pass, nflush);
    (void)hipMemset(bar, 0, 4);
    hipLaunchKernelGGL(probe_k, dim3(nwg), dim3(32), 200 * 1024, 0, (const char*)buf, lat, bar, npg, (uint32_t)pass);
    if (hipDeviceSynchronize()) return -2;
    (void)hipMemcpy(h.data(), lat, h.size() * 2, hipMemcpyDeviceToHost);
    for (uint32_t b = 0; b < npg; b++) {
      const uint32_t c0 = (b * 2654435761u) >> 29;
      for (uint32_t kk = 0; kk < 32; kk++) {
        const uint32_t cls = (c0 + kk + pass) % 8, j = 8 * kk + cls;
        const int fast = h[b * 32 + kk] < 39;
        const int x = wg_dom(j) ^ inpage(probe_off(b, kk, cls, pass));
        votes[b] += fast ? x : 1 - x;
        n[b]++;
      }
    }
  }
  int weak = 0;
  for (uint32_t b = 0; b < npg; b++) {
    phase[b] = 2 * votes[b] > n[b];
    const int m = 2 * votes[b] - n[b];
    weak += (m < 0 ? -m : m) * 2 < n[b];
  }
  (void)hipFree(lat);
  (void)hipFree(bar);
  (void)hipFree(flush);
  return weak;
}
