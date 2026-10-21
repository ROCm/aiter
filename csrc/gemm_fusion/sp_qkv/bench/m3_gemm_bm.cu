// SPDX-License-Identifier: MIT
// M3: standalone split-K FP8 GEMM microbench for the R13 (MP64) card: BM32 (== r18 reference), BM64, BM128 variants of the SAME math, producing the raw FP32 partials P, with a
// byte-compare of every BM64/BM128 P against the r18 BM32 x (M/32)-M-tile reference. SOURCE ONLY: written without a compiler or a GPU (see README-m2-m3.md for the build/run recipe and the unverified list).
//
// Relation to r18 (small/r18/device/sk_gemm_fp8.cu, small/r18/device/layout/sk_traits.hpp): `gemm_body` below is the r18 body with these (and only these) generalisations:
//   * AQMAX (max A instructions per producer wave: 2 -> BM32, 4 -> BM64, 8 -> BM128) replaces the fixed 32-row A stage; A_STAGE_BYTES = AQMAX*2048; a_frag_view has AQMAX*16 rows.
//   * The consumer wave still uses the r18 tiled MMA (TM=2, EM=1, EN=4): one 16x16x128 MFMA per (32-row tile mt, K block), the consumer wave cw owns rows mt*32 + cw*16 .. +15 of tile mt.
//     BM64 = 2 accumulator tiles per consumer wave (mt = 0,1), BM128 = 4. Each output element therefore sees EXACTLY the r18 MFMA sequence (same operands, same K order, accumulation from 0), so P is expected byte-identical.
//   * Dead-row mask per 16-row group: tile (mt, cw) is live iff 2*mt + cw < AQ (AQ = ceil(M/16)); r18's `live_wave` is the AQ <= 2 special case. Dead rows are neither loaded (A rows clamped to M-1) nor stored.
//   * PFK (ring depth) is a template parameter; wait_newer<MB, PFK> handles newer in [0, PFK-2] (r18: PFK == 6 only).
//   * P layout [SK][AQMAX*16][N] (r18: [SK][32][N]); SK is a template parameter (r18: SM_SK 3).
//   * No MORI window / epoch words / SM_STAMP: per-WG wall_clock64 start/end stamps are written to a plain buffer instead.
// UNCOMPILED / UNTESTED.
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
using namespace opus;
#endif

namespace bm {
constexpr int BN = 64, BK = 128, KBLOCKS = 48, K_TOTAL = 6144;
constexpr int THREADS = 256, PRODUCER_WAVES = 2;
constexpr int TM = 2, TN = 1, TK = 1;
}

struct BArgs {
    const void* A; const void* W; float* P; unsigned long long* stamps;
    uint32_t M, N, K, strideA, mp_rows;     // mp_rows = P rows per split (AQMAX*16)
};

#ifdef __HIP_DEVICE_COMPILE__
namespace bmd {
OPUS_D void bar() { asm volatile("" ::: "memory"); __builtin_amdgcn_s_barrier(); asm volatile("" ::: "memory"); }
// vmcnt(newer * MB), newer in [0, PFK-2]
template<int MB, int PFK> OPUS_D void wait_newer(int newer) {
    static_assert(PFK >= 3 && PFK - 2 <= 4, "wait_newer is written for newer <= 4");
    switch (newer) {
        case 0: s_waitcnt_vmcnt(number<0>{}); break;
        case 1: s_waitcnt_vmcnt(number<1 * MB>{}); break;
        case 2: s_waitcnt_vmcnt(number<2 * MB>{}); break;
        case 3: s_waitcnt_vmcnt(number<3 * MB>{}); break;
        default: s_waitcnt_vmcnt(number<4 * MB>{}); break;
    }
}
struct U32x8 { uint32_t w[8]; };
// ---- r18 views, verbatim except the A fragment row count ----
OPUS_D auto w_lane_view() { return opus::make_layout<-1>(opus::make_tuple(2, 64, 16), opus::make_tuple(1024, 16, 1)); }
OPUS_D int w_block_origin(int n16, int kb, int K) { return n16 * 16 * K + kb * 2048; }
OPUS_D auto a_lane_view(int strideA) { return opus::make_layout<-1>(opus::make_tuple(8, 8, 16), opus::make_tuple(strideA, 16, 1)); }
template<int ROWS> OPUS_D auto a_frag_view() { return opus::make_layout<-1>(opus::make_tuple(ROWS, 2, 4), opus::make_tuple(bm::BK, 64, 16)); }
OPUS_D auto b_frag_view() { return opus::make_layout<-1>(opus::make_tuple(4, 2, 2, 2, 16), opus::make_tuple(2048, 1024, 512, 256, 16)); }
template<class Shape> OPUS_D auto lane_coord(const Shape& shape, int lane) {
    return opus::flat_to_coords(lane, opus::make_index_seq<opus::size<Shape>()>{}, shape);
}
OPUS_D auto consumer_wave_coord(int cw) {
    return opus::flat_to_coords(cw, opus::make_index_seq<3>{}, opus::make_tuple(opus::number<bm::TM>{}, opus::number<bm::TN>{}, opus::number<bm::TK>{}));
}
template<class MMA> OPUS_D auto accumulator_register_view(const MMA& mma) {
    using Base = typename MMA::MMA;
    return opus::make_layout<-1>(mma.y_shape_c(),
        opus::make_tuple(opus::number<MMA::expd_n * Base::elem_c>{}, opus::number<Base::pack_c>{}, opus::number<1>{}, opus::number<Base::elem_c>{}));
}
template<int SK> OPUS_D int kb_per_split() { return (bm::KBLOCKS + SK - 1) / SK; }
template<int SK> OPUS_D int kb_begin(int s) { return s * kb_per_split<SK>(); }
template<int SK> OPUS_D int kb_end(int s) { const int e = (s + 1) * kb_per_split<SK>(); return (s == SK - 1 || e > bm::KBLOCKS) ? bm::KBLOCKS : e; }

// AQ = A instructions per producer wave actually used (ceil(M/16)); AQMAX = stage capacity; PFK = ring depth; SK = split-K factor.
template<int AQ, int AQMAX, int PFK, int SK>
OPUS_D void gemm_body(const BArgs& args, unsigned char* stage) {
    static_assert(AQ >= 1 && AQ <= AQMAX, "AQ in [1, AQMAX]");
    constexpr int MB = 4 + AQ;                                   // vmem instructions per producer wave per K block
    constexpr int MT = (AQ + 1) / 2;                             // 32-row accumulator tiles per consumer wave
    constexpr int A_STAGE_BYTES = AQMAX * 2048;                  // AQMAX*16 rows x 128 B
    constexpr int B_STAGE_BYTES = bm::BN * bm::BK;               // 8192
    constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;
    const int tid = thread_id_x();
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const bool producer = wave < bm::PRODUCER_WAVES;
    const int split = block_id_y();
    const int col_base = block_id_x() * bm::BN;
    const int M = static_cast<int>(args.M), N = static_cast<int>(args.N), K = static_cast<int>(args.K);
    const int strideA = static_cast<int>(args.strideA);
    const int kb0 = kb_begin<SK>(split), kb1 = kb_end<SK>(split), L = kb1 - kb0;
    const int wg = block_id_y() * static_cast<int>(gridDim.x) + block_id_x();
    if (tid == 0) args.stamps[wg * 2 + 0] = wall_clock64();

    auto base = make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I);
    auto mma = make_tiled_mma(base, seq<1, 4, 1>{}, seq<bm::TM, bm::TN, bm::TK>{});
    using MMA = decltype(mma);
    typename MMA::vtype_c accum[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) accum[t] = typename MMA::vtype_c{0};

    auto ga = make_gmem(static_cast<const fp8_t*>(args.A), static_cast<unsigned int>(static_cast<unsigned long long>(M - 1) * args.strideA + K));
    auto gb = make_gmem(static_cast<const fp8_t*>(args.W), static_cast<unsigned int>(static_cast<unsigned long long>(N) * K));
    const auto u_wl = w_lane_view();

    if (producer) {
        const int n16_0 = col_base / 16;
        int b_src[4], b_dst[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int f = wave * 4 + q, nb = f >> 1, half = f & 1;
            b_src[q] = w_block_origin(n16_0 + nb, 0, K) + static_cast<int>(u_wl(half, lane, 0));
            b_dst[q] = A_STAGE_BYTES + nb * 2048 + half * 1024;
        }
        int a_src[AQ], a_dst[AQ];
        #pragma unroll
        for (int j = 0; j < AQ; ++j) {
            const int a = wave * AQ + j;
            int r = a * 8 + (lane >> 3);
            r = (r < M) ? r : (M - 1);
            a_src[j] = r * strideA + (lane & 7) * 16;
            a_dst[j] = a * 1024;
        }
        auto issue = [&](int kb, int slot) {
            unsigned char* const st = stage + slot * STAGE_BYTES;
            #pragma unroll
            for (int j = 0; j < AQ; ++j) ga.template async_load<16>(st + a_dst[j], a_src[j] + kb * bm::BK);
            #pragma unroll
            for (int q = 0; q < 4; ++q) gb.template async_load<16>(st + b_dst[q], b_src[q] + kb * 2048);
        };
        const int pro = (L < PFK - 1) ? L : (PFK - 1);
        for (int p = 0; p < pro; ++p) issue(kb0 + p, p);
        for (int i = 0; i < L; ++i) {
            const int remaining = L - 1 - i;
            const int newer = (remaining < PFK - 2) ? remaining : (PFK - 2);
            wait_newer<MB, PFK>(newer);
            bar();
            const int nxt = i + PFK - 1;
            if (nxt < L) issue(kb0 + nxt, nxt % PFK);
        }
        s_waitcnt_vmcnt(number<0>{});
    } else {
        const int cw = wave - bm::PRODUCER_WAVES;                 // 0: rows 0-15 of every 32-row tile, 1: rows 16-31
        const auto wc = consumer_wave_coord(cw);
        const auto lc = lane_coord(base.p_shape_c(), lane);
        const auto pc = make_tuple(get<0>(wc), get<0>(lc), get<1>(wc), get<1>(lc));
        const auto u_cm = partition_layout_c(mma, make_tuple(1_I, 0_I), pc);
        const auto u_cn = partition_layout_c(mma, make_tuple(0_I, 1_I), pc);
        const auto u_acc = accumulator_register_view(mma);
        const int l16 = lane & 15, g = lane >> 4;
        const auto u_af = a_frag_view<AQMAX * 16>();
        const auto u_bf = b_frag_view();
        int a_off0[MT], a_off1[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int a_row = t * 32 + cw * 16 + l16;
            a_off0[t] = static_cast<int>(u_af(a_row, 0, g)); a_off1[t] = static_cast<int>(u_af(a_row, 1, g));
        }
        int b_off0[4], b_off1[4];
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            b_off0[nb] = A_STAGE_BYTES + static_cast<int>(u_bf(nb, 0, g >> 1, g & 1, l16));
            b_off1[nb] = A_STAGE_BYTES + static_cast<int>(u_bf(nb, 1, g >> 1, g & 1, l16));
        }
        auto lds = smem<fp8_t>(stage);
        for (int i = 0; i < L; ++i) {
            bar();
            const int sb = (i % PFK) * STAGE_BYTES;
            if (2 * 0 + cw < AQ) {                                  // some tile live for this wave (tile 0 is the lowest row group)
                typename MMA::vtype_a va[MT]; typename MMA::vtype_b vb;
                #pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (2 * t + cw < AQ) {
                        auto s0 = lds.template load<16>(sb + a_off0[t]); auto s1 = lds.template load<16>(sb + a_off1[t]);
                        U32x8 tt; __builtin_memcpy(&tt.w[0], &s0, 16); __builtin_memcpy(&tt.w[4], &s1, 16); __builtin_memcpy(&va[t], &tt, 32);
                    }
                }
                {
                    U32x8 tt[4];
                    #pragma unroll
                    for (int nb = 0; nb < 4; ++nb) {
                        auto s0 = lds.template load<16>(sb + b_off0[nb]); auto s1 = lds.template load<16>(sb + b_off1[nb]);
                        __builtin_memcpy(&tt[nb].w[0], &s0, 16); __builtin_memcpy(&tt[nb].w[4], &s1, 16);
                    }
                    __builtin_memcpy(&vb, &tt[0], 128);
                }
                s_waitcnt_lgkmcnt(number<0>{});
                #pragma unroll
                for (int t = 0; t < MT; ++t)
                    if (2 * t + cw < AQ) accum[t] = mma(va[t], vb, accum[t], 127, 127, number<0>{}, number<0>{});
            }
        }
        const unsigned long long prow = args.mp_rows;
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            static_ford(mma.y_shape_c(), [&](auto im, auto rept, auto item, auto jn) {
                const auto y = make_tuple(im, rept, item, jn);
                const int m = t * 32 + u_cm(y), n = col_base + u_cn(y);
                if (m < M && n < N) {
                    const int acc_idx = static_cast<int>(u_acc(y));
                    args.P[(static_cast<unsigned long long>(split) * prow + m) * N + n] = accum[t][acc_idx];
                }
            });
        }
        s_waitcnt_vmcnt(number<0>{});
    }
    if (tid == 0) args.stamps[wg * 2 + 1] = wall_clock64();
}
} // namespace bmd

template<int AQMAX, int PFK, int SK>
__global__ __launch_bounds__(256) void gemm_bm(BArgs args) {
#if defined(__gfx950__)
    constexpr int LDS_BYTES = PFK * (AQMAX * 2048 + 8192);
    static_assert(LDS_BYTES * 2 <= 163840, "2 WG/CU LDS budget");
    if (args.K != 128u * bm::KBLOCKS || args.M == 0u || args.M > static_cast<uint32_t>(AQMAX * 16) || args.mp_rows != static_cast<uint32_t>(AQMAX * 16)) return;
    __shared__ __align__(16) unsigned char stage[LDS_BYTES];
    const int aq = static_cast<int>((args.M + 15u) / 16u);
    switch (aq) {   // WG-uniform
        case 1: bmd::gemm_body<1, AQMAX, PFK, SK>(args, stage); break;
        case 2: bmd::gemm_body<2, AQMAX, PFK, SK>(args, stage); break;
        case 3: if constexpr (AQMAX >= 3) bmd::gemm_body<(AQMAX >= 3 ? 3 : 1), AQMAX, PFK, SK>(args, stage); break;
        case 4: if constexpr (AQMAX >= 4) bmd::gemm_body<(AQMAX >= 4 ? 4 : 1), AQMAX, PFK, SK>(args, stage); break;
        case 5: if constexpr (AQMAX >= 5) bmd::gemm_body<(AQMAX >= 5 ? 5 : 1), AQMAX, PFK, SK>(args, stage); break;
        case 6: if constexpr (AQMAX >= 6) bmd::gemm_body<(AQMAX >= 6 ? 6 : 1), AQMAX, PFK, SK>(args, stage); break;
        case 7: if constexpr (AQMAX >= 7) bmd::gemm_body<(AQMAX >= 7 ? 7 : 1), AQMAX, PFK, SK>(args, stage); break;
        default: if constexpr (AQMAX >= 8) bmd::gemm_body<(AQMAX >= 8 ? 8 : 1), AQMAX, PFK, SK>(args, stage); break;
    }
#else
    __builtin_trap();
#endif
}
#else   // host pass: kernel templates need only a declaration of the same signature
template<int AQMAX, int PFK, int SK> __global__ void gemm_bm(BArgs) {}
#endif

// ------------------------------------------------------------------ host ------------------------------------------------------------------
#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Variant { const char* name; int aqmax; int pfk; };
template<int AQMAX, int PFK, int SK> static void launch(const BArgs& a, int ntiles, hipStream_t s) {
    hipLaunchKernelGGL((gemm_bm<AQMAX, PFK, SK>), dim3(ntiles, SK, 1), dim3(256, 1, 1), 0, s, a);
}
static void launch_variant(int aqmax, int pfk, int sk, const BArgs& a, int ntiles, hipStream_t s) {
    if (sk == 3) {
        if (aqmax == 2 && pfk == 6) return launch<2, 6, 3>(a, ntiles, s);
        if (aqmax == 4 && pfk == 4) return launch<4, 4, 3>(a, ntiles, s);
        if (aqmax == 4 && pfk == 5) return launch<4, 5, 3>(a, ntiles, s);
        if (aqmax == 8 && pfk == 3) return launch<8, 3, 3>(a, ntiles, s);
    } else if (sk == 2) {
        if (aqmax == 2 && pfk == 6) return launch<2, 6, 2>(a, ntiles, s);
        if (aqmax == 4 && pfk == 4) return launch<4, 4, 2>(a, ntiles, s);
        if (aqmax == 4 && pfk == 5) return launch<4, 5, 2>(a, ntiles, s);
        if (aqmax == 8 && pfk == 3) return launch<8, 3, 2>(a, ntiles, s);
    }
    std::fprintf(stderr, "unsupported variant aqmax=%d pfk=%d sk=%d\n", aqmax, pfk, sk); std::exit(3);
}

static void fill_fp8(std::vector<uint8_t>& v, uint32_t seed) {      // no NaN encodings (e4m3fn NaN = S.1111.111): clear exponent MSB and bit 3
    uint32_t x = seed * 2654435761u + 1u;
    for (auto& b : v) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; b = static_cast<uint8_t>(x & 0xB7u); }
}

int main(int argc, char** argv) {
    int M = 64, N = 9216, iters = 200, warm = 20, sk = 3, cold = 0, dev = 0; uint32_t seed = 12345;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char* { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "--m") M = std::atoi(nxt()); else if (a == "--n") N = std::atoi(nxt()); else if (a == "--iters") iters = std::atoi(nxt());
        else if (a == "--warm") warm = std::atoi(nxt()); else if (a == "--sk") sk = std::atoi(nxt()); else if (a == "--cold") cold = std::atoi(nxt());
        else if (a == "--device") dev = std::atoi(nxt()); else if (a == "--seed") seed = static_cast<uint32_t>(std::atoi(nxt()));
        else if (a == "--only") only.push_back(nxt());
        else { std::fprintf(stderr, "usage: m3_gemm_bm [--m 1..128] [--n 9216|9856] [--sk 2|3] [--iters I] [--warm W] [--cold 0|1 (flush L2 between launches)] [--device D] [--only bm64_pfk4 ...]\n"); return 1; }
    }
    if (M < 1 || M > 128 || N % 64 != 0) { std::fprintf(stderr, "bad M/N\n"); return 1; }
    HIPCK(hipSetDevice(dev));
    const int K = bm::K_TOTAL, ntiles = N / 64;
    std::vector<uint8_t> hA(static_cast<size_t>(M) * K), hW(static_cast<size_t>(N) * K);
    fill_fp8(hA, seed); fill_fp8(hW, seed + 7);
    uint8_t *dA = nullptr, *dW = nullptr; unsigned long long* dstamp = nullptr; uint8_t* flush = nullptr;
    HIPCK(hipMalloc(&dA, hA.size())); HIPCK(hipMalloc(&dW, hW.size()));
    HIPCK(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice)); HIPCK(hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice));
    const int maxwg = ntiles * sk;
    HIPCK(hipMalloc(&dstamp, sizeof(unsigned long long) * 2 * maxwg));
    const size_t flush_bytes = 512ull << 20; if (cold) HIPCK(hipMalloc(&flush, flush_bytes));
    hipStream_t st; HIPCK(hipStreamCreate(&st));
    hipEvent_t e0, e1; HIPCK(hipEventCreate(&e0)); HIPCK(hipEventCreate(&e1));

    // ---------------- reference: r18 BM32 (AQMAX 2, PFK 6) once per 32-row M tile ----------------
    const int tiles = (M + 31) / 32;
    std::vector<std::vector<float>> ref(tiles);
    for (int t = 0; t < tiles; ++t) {
        const int mt = std::min(32, M - 32 * t);
        float* dP = nullptr; const size_t pn = static_cast<size_t>(sk) * 32 * N;
        HIPCK(hipMalloc(&dP, pn * 4)); HIPCK(hipMemset(dP, 0, pn * 4));
        BArgs a{dA + static_cast<size_t>(32) * t * K, dW, dP, dstamp, static_cast<uint32_t>(mt), static_cast<uint32_t>(N), static_cast<uint32_t>(K), static_cast<uint32_t>(K), 32u};
        launch_variant(2, 6, sk, a, ntiles, st); HIPCK(hipStreamSynchronize(st));
        ref[t].resize(pn); HIPCK(hipMemcpy(ref[t].data(), dP, pn * 4, hipMemcpyDeviceToHost)); HIPCK(hipFree(dP));
    }
    std::printf("# M3 gemm microbench: M=%d N=%d K=%d SK=%d ntiles=%d WGs=%d (reference = BM32/PFK6 x %d M-tile launch(es), the r18 math)\n", M, N, K, sk, ntiles, maxwg, tiles);
    std::printf("variant\tAQMAX\tPFK\tLDS_KiB\tP_bytes\tevent_us_med\tevent_us_min\tstamp_span_us_med\tP_byte_equal_vs_r18_ref\tfirst_diff\n");

    std::vector<Variant> vars = { {"bm32_pfk6", 2, 6}, {"bm64_pfk4", 4, 4}, {"bm64_pfk5", 4, 5}, {"bm128_pfk3", 8, 3} };
    for (const auto& v : vars) {
        if (!only.empty() && std::find(only.begin(), only.end(), std::string(v.name)) == only.end()) continue;
        const int rows = v.aqmax * 16;
        if (M > rows) { std::printf("%s\t%d\t%d\t-\t-\t-\t-\t-\tSKIP (M > %d)\t-\n", v.name, v.aqmax, v.pfk, rows); continue; }
        float* dP = nullptr; const size_t pn = static_cast<size_t>(sk) * rows * N;
        HIPCK(hipMalloc(&dP, pn * 4)); HIPCK(hipMemset(dP, 0, pn * 4));
        BArgs a{dA, dW, dP, dstamp, static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K), static_cast<uint32_t>(K), static_cast<uint32_t>(rows)};
        std::vector<float> ev; std::vector<double> spans;
        for (int it = 0; it < warm + iters; ++it) {
            if (cold) HIPCK(hipMemsetAsync(flush, it & 0xFF, flush_bytes, st));
            HIPCK(hipEventRecord(e0, st)); launch_variant(v.aqmax, v.pfk, sk, a, ntiles, st); HIPCK(hipEventRecord(e1, st)); HIPCK(hipStreamSynchronize(st));
            if (it >= warm) {
                float ms = 0; HIPCK(hipEventElapsedTime(&ms, e0, e1)); ev.push_back(ms * 1000.0f);
                std::vector<unsigned long long> hs(2 * maxwg); HIPCK(hipMemcpy(hs.data(), dstamp, hs.size() * 8, hipMemcpyDeviceToHost));
                unsigned long long mn = ~0ull, mx = 0; for (int w = 0; w < maxwg; ++w) { mn = std::min(mn, hs[2 * w]); mx = std::max(mx, hs[2 * w + 1]); }
                spans.push_back(static_cast<double>(mx - mn) / 100.0);        // wall_clock64 ticks at 100 MHz -> us
            }
        }
        // P byte-compare vs the r18 reference (live rows only): P64[s][t*32 + r][n] == ref_t[s][r][n]
        std::vector<float> hP(pn); HIPCK(hipMemcpy(hP.data(), dP, pn * 4, hipMemcpyDeviceToHost));
        bool eq = true; long long first = -1;
        for (int s = 0; s < sk && eq; ++s) for (int m = 0; m < M && eq; ++m) {
            const int t = m / 32, r = m % 32;
            const float* x = &hP[(static_cast<size_t>(s) * rows + m) * N]; const float* y = &ref[t][(static_cast<size_t>(s) * 32 + r) * N];
            if (std::memcmp(x, y, static_cast<size_t>(N) * 4) != 0) { eq = false; for (int n = 0; n < N; ++n) if (std::memcmp(&x[n], &y[n], 4) != 0) { first = (static_cast<long long>(s) * 1000 + m) * 100000LL + n; break; } }
        }
        std::sort(ev.begin(), ev.end()); std::sort(spans.begin(), spans.end());
        const int lds = v.pfk * (v.aqmax * 2048 + 8192) / 1024;
        std::printf("%s\t%d\t%d\t%d\t%zu\t%.3f\t%.3f\t%.3f\t%s\t%lld\n", v.name, v.aqmax, v.pfk, lds, pn * 4, ev[ev.size() / 2], ev.front(), spans[spans.size() / 2], eq ? "EQUAL" : "DIFFERENT", first);
        HIPCK(hipFree(dP));
    }
    return 0;
}
