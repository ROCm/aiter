// SPDX-License-Identifier: MIT
// M2: CU-push microbench for the R13 (MP64) card: how long do G workgroups of T threads need to push B bytes each to peer GPUs and release, as a function of
// bytes per WG, WG count, threads per WG, releases per WG and number of destinations. SOURCE ONLY (no compiler, no GPU when written); see README-m2-m3.md.
//
// Idioms copied from small/r18/device/ran_kernel.cu so the numbers transfer:
//   * data: one 16 B vector per lane per store, opus gmem store<4> with aux 17 (= SC0|SC1, system-coherent write-through) through a buffer resource whose base is the wave-uniform destination base  (push16, ran l.315-318);
//   * retire: every wave `s_waitcnt vmcnt(0)` (stores acknowledged) -> workgroup barrier;
//   * release: thread 0 fence(release, system) -> `s_waitcnt vmcnt(0)` -> relaxed system-scope RMW (fetch_add 1) on a flag word of the destination (sys_add_release, ran l.657-661);
//   * WG -> destination: contiguous blocks (WG w pushes to destination (w * D) / G), like the r18 role table (8 WGs per destination at 24 WGs / 3 destinations); XCC of a WG = wgid % 8 (dispatch rule used by the RAN kernel).
// One process, GPU 0 = sender, GPUs 1..D = receivers, peer access enabled from 0. The flag chain, polling and the compute are NOT modelled: this isolates store retire and release cost.
// Per-WG wall_clock64 stamps (100 MHz): t0 = WG entry, t1 = after all stores retired + barrier, t2 = after the last release RMW returned.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#ifdef __HIP_DEVICE_COMPILE__
#include <opus/opus.hpp>
#endif

struct PushArgs {
    unsigned char* dst[4];              // receiver buffers (device pointers valid on the sender via peer access), size G * bytes_per_wg each
    unsigned long long* flag[4];        // receiver flag words (one per destination)
    unsigned long long* stamps;         // [G][4] on the sender: t0, t1, t2, (xcc)
    uint32_t bytes_per_wg, rounds, releases, ndst, mode;   // mode 0 = SC0|SC1 write-through (r18 push16), 1 = plain store, 2 = non-temporal
    uint32_t epoch;
};

#ifdef __HIP_DEVICE_COMPILE__
__device__ __forceinline__ void sys_add_release(void* p, unsigned long long v) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __hip_atomic_fetch_add(reinterpret_cast<unsigned long long*>(p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ __launch_bounds__(512) void m2_push(PushArgs a) {
#if defined(__gfx950__)
    const uint32_t G = gridDim.x, w = blockIdx.x, T = blockDim.x, tid = threadIdx.x;
    const uint32_t d = (w * a.ndst) / G;                         // destination slot of this WG
    __shared__ unsigned long long s_t0;
    if (tid == 0) s_t0 = wall_clock64();
    unsigned char* const base = a.dst[d] + static_cast<size_t>(w) * a.bytes_per_wg;       // this WG's private region on the destination
    // 16 B per lane per store; the lanes of a wave cover 1 KiB contiguous (like the 16-lane x 8 bf16 rows of the RAN push)
    const uint32_t vec = a.bytes_per_wg / 16u;
    for (uint32_t r = 0; r < a.rounds; ++r) {
        for (uint32_t i = tid; i < vec; i += T) {
            opus::vector_t<opus::u32_t, 4> x;
            const uint32_t seed = (a.epoch * 0x9E3779B1u) ^ (w * 0x85EBCA6Bu) ^ (i * 0xC2B2AE35u) ^ r;
            x[0] = seed; x[1] = seed + 1u; x[2] = seed + 2u; x[3] = seed + 3u;
            if (a.mode == 0) {
                auto g = opus::make_gmem(reinterpret_cast<const opus::u32_t*>(base), a.bytes_per_wg);
                g.template store<4>(x, static_cast<int>(i * 4u), 0, opus::number<17>{});    // u32 units; aux = SC0|SC1 (push16)
            } else if (a.mode == 1) {
                reinterpret_cast<opus::vector_t<opus::u32_t, 4>*>(base)[i] = x;
            } else {
                __builtin_nontemporal_store(x, reinterpret_cast<opus::vector_t<opus::u32_t, 4>*>(base) + i);
            }
        }
    }
    __builtin_amdgcn_s_waitcnt(0x0);                             // EVERY wave: stores acknowledged
    __syncthreads();
    unsigned long long t1 = 0;
    if (tid == 0) t1 = wall_clock64();
    if (tid == 0) {
        for (uint32_t k = 0; k < a.releases; ++k) sys_add_release(a.flag[d], 1ull);          // `releases` queued releases per WG (r18: 1, straddlers 2)
        const unsigned long long t2 = wall_clock64();
        unsigned long long* s = a.stamps + static_cast<size_t>(w) * 4;
        s[0] = s_t0; s[1] = t1; s[2] = t2; s[3] = static_cast<unsigned long long>(w & 7u);
    }
#else
    __builtin_trap();
#endif
}
#else
__global__ void m2_push(PushArgs) {}
#endif

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Row { int G, T, rel, ndst, mode, rounds; uint32_t bytes; };

int main(int argc, char** argv) {
    std::vector<uint32_t> bytes = {32768}; std::vector<int> wgs = {24}, thr = {512}, rels = {1}, ndsts = {3}, modes = {0};
    int rounds = 1, reps = 200, warm = 20;
    auto parse = [](const char* s, auto& out) { out.clear(); std::string x(s); size_t p = 0; while (p <= x.size()) { size_t q = x.find(',', p); if (q == std::string::npos) q = x.size(); out.push_back(static_cast<typename std::decay<decltype(out)>::type::value_type>(std::atoll(x.substr(p, q - p).c_str()))); p = q + 1; } };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto nxt = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "--bytes") parse(nxt(), bytes); else if (a == "--wgs") parse(nxt(), wgs); else if (a == "--threads") parse(nxt(), thr); else if (a == "--rel") parse(nxt(), rels);
        else if (a == "--ndst") parse(nxt(), ndsts); else if (a == "--mode") parse(nxt(), modes); else if (a == "--rounds") rounds = std::atoi(nxt());
        else if (a == "--reps") reps = std::atoi(nxt()); else if (a == "--warm") warm = std::atoi(nxt());
        else { std::fprintf(stderr, "usage: m2_cu_push [--bytes B,..] [--wgs G,..] [--threads 256,512] [--rel 0,1,2,3] [--ndst 1,2,3] [--mode 0|1|2 (sc0sc1|plain|nt)] [--rounds R] [--reps N] [--warm W]  (lists are comma-separated; the sweep is their cartesian product)\n"); return 1; }
    }
    int ngpu = 0; HIPCK(hipGetDeviceCount(&ngpu));
    int maxd = *std::max_element(ndsts.begin(), ndsts.end());
    if (ngpu < 1 + maxd) { std::fprintf(stderr, "need %d GPUs for --ndst %d (have %d)\n", 1 + maxd, maxd, ngpu); return 1; }
    HIPCK(hipSetDevice(0));
    for (int d = 1; d <= maxd; ++d) { int can = 0; HIPCK(hipDeviceCanAccessPeer(&can, 0, d)); if (!can) { std::fprintf(stderr, "GPU0 cannot access GPU%d\n", d); return 1; } hipError_t e = hipDeviceEnablePeerAccess(d, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIPCK(e); }
    const uint32_t maxbytes = *std::max_element(bytes.begin(), bytes.end()); const int maxG = *std::max_element(wgs.begin(), wgs.end());
    unsigned char* dbuf[4] = {}; unsigned long long* dflag[4] = {}; unsigned long long* dstamp = nullptr;
    for (int d = 1; d <= maxd; ++d) { HIPCK(hipSetDevice(d)); HIPCK(hipMalloc(&dbuf[d - 1], static_cast<size_t>(maxbytes) * maxG)); HIPCK(hipMalloc(&dflag[d - 1], 64)); }
    HIPCK(hipSetDevice(0)); HIPCK(hipMalloc(&dstamp, sizeof(unsigned long long) * 4 * maxG));
    hipStream_t st; HIPCK(hipStreamCreate(&st)); hipEvent_t e0, e1; HIPCK(hipEventCreate(&e0)); HIPCK(hipEventCreate(&e1));

    std::printf("# M2 CU-push: sender GPU0 -> receivers GPU1..D, stores per mode, release = fence(release,system)+vmcnt(0)+relaxed system RMW (r18 sys_add_release). times in us; stamps = wall_clock64 @100 MHz\n");
    std::printf("bytes_per_wg\twgs\tthreads\treleases\tndst\tmode\ttotal_KiB\tevent_us_med\tretire_us_med\trelease_us_med\tgbps_event\tflag_ok\tdata_ok\n");
    uint32_t epoch = 0;
    for (auto B : bytes) for (auto G : wgs) for (auto T : thr) for (auto R : rels) for (auto D : ndsts) for (auto mode : modes) {
        if (B % 16 != 0 || G < D) continue;
        PushArgs a{}; for (int d = 0; d < 4; ++d) { a.dst[d] = dbuf[d]; a.flag[d] = dflag[d]; } a.stamps = dstamp;
        a.bytes_per_wg = B; a.rounds = rounds; a.releases = R; a.ndst = D; a.mode = mode;
        std::vector<float> ev, ret, rel; bool flag_ok = true, data_ok = true;
        for (int it = 0; it < warm + reps; ++it) {
            for (int d = 1; d <= D; ++d) { HIPCK(hipSetDevice(d)); HIPCK(hipMemset(dflag[d - 1], 0, 64)); HIPCK(hipDeviceSynchronize()); }
            HIPCK(hipSetDevice(0)); a.epoch = ++epoch;
            HIPCK(hipEventRecord(e0, st)); hipLaunchKernelGGL(m2_push, dim3(G), dim3(T), 0, st, a); HIPCK(hipEventRecord(e1, st)); HIPCK(hipStreamSynchronize(st));
            if (it < warm) continue;
            float ms = 0; HIPCK(hipEventElapsedTime(&ms, e0, e1)); ev.push_back(ms * 1000.0f);
            std::vector<unsigned long long> s(4 * G); HIPCK(hipMemcpy(s.data(), dstamp, s.size() * 8, hipMemcpyDeviceToHost));
            unsigned long long t0 = ~0ull, t1 = 0, t2 = 0; for (int w = 0; w < G; ++w) { t0 = std::min(t0, s[4 * w]); t1 = std::max(t1, s[4 * w + 1]); t2 = std::max(t2, s[4 * w + 2]); }
            ret.push_back(static_cast<float>(t1 - t0) / 100.0f); rel.push_back(static_cast<float>(t2 - t1) / 100.0f);
            for (int d = 1; d <= D; ++d) {                       // flag value must equal (#WGs of this destination) * releases
                unsigned long long fv = 0; HIPCK(hipMemcpy(&fv, dflag[d - 1], 8, hipMemcpyDeviceToHost));
                int cnt = 0; for (int w = 0; w < G; ++w) if (static_cast<int>((static_cast<long long>(w) * D) / G) == d - 1) ++cnt;
                if (fv != static_cast<unsigned long long>(cnt) * R) flag_ok = false;
            }
        }
        {   // data check of the last launch: the last round of WG 0's first and last vector must hold the expected pattern (rounds > 1 overwrites the same region)
            const uint32_t d = 0; std::vector<unsigned char> h(B); HIPCK(hipMemcpy(h.data(), dbuf[d], B, hipMemcpyDeviceToHost));
            auto exp = [&](uint32_t w, uint32_t i, uint32_t r, uint32_t k) { return ((epoch * 0x9E3779B1u) ^ (w * 0x85EBCA6Bu) ^ (i * 0xC2B2AE35u) ^ r) + k; };
            for (uint32_t i : {0u, B / 16u - 1u}) { uint32_t w4[4]; std::memcpy(w4, &h[i * 16], 16); for (uint32_t k = 0; k < 4; ++k) if (w4[k] != exp(0, i, rounds - 1, k)) data_ok = false; }
        }
        std::sort(ev.begin(), ev.end()); std::sort(ret.begin(), ret.end()); std::sort(rel.begin(), rel.end());
        const double total_kib = static_cast<double>(B) * G / 1024.0;
        std::printf("%u\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.3f\t%.3f\t%.3f\t%.2f\t%d\t%d\n", B, G, T, R, D, mode, total_kib, ev[ev.size() / 2], ret[ret.size() / 2], rel[rel.size() / 2],
                    (static_cast<double>(B) * G * rounds) / (ev[ev.size() / 2] * 1e-6) / 1e9, flag_ok ? 1 : 0, data_ok ? 1 : 0);
    }
    return 0;
}
