// SPDX-License-Identifier: MIT
// SM-SK2-RAN-r3 FP8 split-K GEMM (GEMM math unchanged from r2; r3 only splits the STAMPS words producer/consumer). UNCOMPILED / UNTESTED. Numerics identical to r1: split s owns contiguous K128 blocks
// [kb_begin,kb_end); FP32 accumulation from 0 in K order; raw FP32 partial P[s][m][n]; no scale, no BF16.
// G1 PFK=6 LDS ring; producer waves keep PFK-1 K-blocks in flight with counted vmcnt waits (aiter flatmm split-K structure:
//    producer waves 0-1, consumer waves 2-3, one s_barrier per K block, vmcnt(0) only through the tail).
// G2 gmem::async_load (buffer_load ... lds): no VGPR round trip.
// G3 W staged in PHYSICAL shuffle order: one band x K128 = 2 KiB contiguous = 2 wave instructions of 1 KiB.
// G4 dead A rows (m>=M) neither loaded nor staged: M<=16 -> AQ=1, M>16 -> AQ=2. The consumer wave owning rows 16..31 skips its
//    MFMA when M<=16 (its outputs are masked). Source rows clamped to M-1 (never OOB).
// K permutation: A and B fragments are each two 16 B LDS segments 64 K apart (+g*16 within a segment); the SAME permutation for
//    A and B, so the MFMA K-sum is exact for any internal pairing (unverified vs hardware convention -> GEMM unit falsifier).
#include "sm_abi_v2.h"
#include "stamps.hpp"
#include <opus/hip_minimal.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include <opus/opus.hpp>
#include "layout/sk_traits.hpp"
using namespace opus;

namespace smg {
OPUS_D void bar() { asm volatile("" ::: "memory"); __builtin_amdgcn_s_barrier(); asm volatile("" ::: "memory"); }
// vmcnt(newer * MB), newer in [0, PFK-2]; PFK-2 == 4
template<int MB> OPUS_D void wait_newer(int newer) {
    switch (newer) {
        case 0: s_waitcnt_vmcnt(number<0>{}); break;
        case 1: s_waitcnt_vmcnt(number<1 * MB>{}); break;
        case 2: s_waitcnt_vmcnt(number<2 * MB>{}); break;
        case 3: s_waitcnt_vmcnt(number<3 * MB>{}); break;
        default: s_waitcnt_vmcnt(number<4 * MB>{}); break;
    }
}
static_assert(sks::PFK == 6, "wait_newer is written for PFK-2 == 4");

struct U32x8 { uint32_t w[8]; };     // 32 fp8 = one MFMA A or B operand (no HIP vector types under RTC)

template<int AQ>
OPUS_D void gemm_body(const SmGemmArgsV1& args, unsigned char* stage) {
    constexpr int MB = 4 + AQ;                         // vmem instructions per producer wave per K block
    const int tid = thread_id_x();
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const bool producer = wave < sks::PRODUCER_WAVES;
    const int split = block_id_y();
    const int col_base = block_id_x() * sks::BN;
    const int M = static_cast<int>(args.M), N = static_cast<int>(args.N), K = static_cast<int>(args.K);
    const int strideA = static_cast<int>(args.strideA);
    const int kb0 = sks::kb_begin(split), kb1 = sks::kb_end(split), L = kb1 - kb0;
    const int wg = block_id_y() * static_cast<int>(SM_GEMM_NTILES) + block_id_x();     // r18: real grid width (144 with SM_REUSE_NO_IDX); stamps index only
    uint64_t* const rec = SM_STAMP_REC(args.dbg, SM_DBG_RAN_MAXWG, wg);
    if (tid == 0) { SM_STAMP(rec, 0); SM_STAMP_HWID(rec, 16); SM_STAMP_XCC(rec, 58); }

    auto base = make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I);
    auto mma = make_tiled_mma(base, seq<sks::EM, sks::EN, sks::EK>{}, seq<sks::TM, sks::TN, sks::TK>{});
    using MMA = decltype(mma);
    typename MMA::vtype_c accum{0};

    auto ga = make_gmem(static_cast<const fp8_t*>(args.A),
        static_cast<unsigned int>(static_cast<unsigned long long>(M - 1) * args.strideA + K));
    auto gb = make_gmem(static_cast<const fp8_t*>(args.W), static_cast<unsigned int>(static_cast<unsigned long long>(N) * K));
    const auto u_wl = sks::w_lane_view();                  // (half, lane, byte)
    const auto u_al = sks::a_lane_view(strideA);           // (row8, chunk, byte)

    if (producer) {
        // ---- per-lane source offsets (kb-independent parts) ----
        const int n16_0 = col_base / 16;
        // B: instruction q in [0,4): flat f = wave*4 + q -> band nb = f/2, half = f%2
        int b_src[4], b_dst[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int f = wave * 4 + q, nb = f >> 1, half = f & 1;
            b_src[q] = sks::w_block_origin(n16_0 + nb, 0, K) + static_cast<int>(u_wl(half, lane, 0));   // + kb*2048 later
            b_dst[q] = sks::A_STAGE_BYTES + nb * 2048 + half * 1024;                                       // wave-uniform LDS dst
        }
        // A: instruction j in [0,AQ): flat a = wave*AQ + j -> rows 8a..8a+7; lane -> (row8 = lane/8, chunk = lane%8)
        int a_src[AQ], a_dst[AQ];
        #pragma unroll
        for (int j = 0; j < AQ; ++j) {
            const int a = wave * AQ + j;
            int r = a * 8 + (lane >> 3);
            r = (r < M) ? r : (M - 1);                                                                      // clamp: never OOB
            a_src[j] = r * strideA + (lane & 7) * 16;                                                       // + kb*128 later
            a_dst[j] = a * 1024;
        }
        auto issue = [&](int kb, int slot) {
            unsigned char* const st = stage + slot * sks::STAGE_BYTES;
            #pragma unroll
            for (int j = 0; j < AQ; ++j)
                ga.template async_load<16>(st + a_dst[j], a_src[j] + kb * sks::BK);
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                gb.template async_load<16>(st + b_dst[q], b_src[q] + kb * 2048);
        };
        // prologue: PFK-1 blocks in flight (bounded by L)
        const int pro = (L < sks::PFK - 1) ? L : (sks::PFK - 1);
        for (int p = 0; p < pro; ++p) issue(kb0 + p, p);
        for (int i = 0; i < L; ++i) {
            const int remaining = L - 1 - i;                                       // blocks issued after block i so far
            const int newer = (remaining < sks::PFK - 2) ? remaining : (sks::PFK - 2);
            wait_newer<MB>(newer);                                                 // block i complete (counted, FIFO issue)
            bar();                                                                 // R_i: block i visible; slot (i-1)%PFK retired by consumers
            if (i == 0 && tid == 0) SM_STAMP(rec, 1);
            const int nxt = i + sks::PFK - 1;
            if (nxt < L) issue(kb0 + nxt, nxt % sks::PFK);
        }
        s_waitcnt_vmcnt(number<0>{});
        if (tid == 0) SM_STAMP(rec, 2);
    } else {
        const int cw = wave - sks::PRODUCER_WAVES;                                // 0: rows 0-15, 1: rows 16-31
        const auto wc = sks::consumer_wave_coord(cw);
        const auto lc = sks::lane_coord(base.p_shape_c(), lane);
        const auto pc = make_tuple(get<0>(wc), get<0>(lc), get<1>(wc), get<1>(lc));
        const auto u_cm = partition_layout_c(mma, make_tuple(1_I, 0_I), pc);
        const auto u_cn = partition_layout_c(mma, make_tuple(0_I, 1_I), pc);
        const auto u_acc = sks::accumulator_register_view(mma);
        const bool live_wave = (AQ == 2) || (cw == 0);                            // M<=16: rows 16..31 wave idles
        // LDS fragment offsets (bytes inside a stage). A/B lane decode: l16 = lane%16 (row / n), g = lane/16 (K group 0..3).
        const int l16 = lane & 15, g = lane >> 4;
        const auto u_af = sks::a_frag_view();                                     // (row, seg, g)
        const auto u_bf = sks::b_frag_view();                                     // (nb, seg, g/2, g%2, n16)
        const int a_row = cw * 16 + l16;
        const int a_off0 = static_cast<int>(u_af(a_row, 0, g)), a_off1 = static_cast<int>(u_af(a_row, 1, g));
        int b_off0[4], b_off1[4];
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            b_off0[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 0, g >> 1, g & 1, l16));
            b_off1[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 1, g >> 1, g & 1, l16));
        }
        auto lds = smem<fp8_t>(stage);
        for (int i = 0; i < L; ++i) {
            bar();                                                                // R_i
            if (i == 0 && tid == 128) SM_STAMP(rec, 5);
            if (live_wave) {
                const int sb = (i % sks::PFK) * sks::STAGE_BYTES;
                typename MMA::vtype_a va; typename MMA::vtype_b vb;
                {
                    auto s0 = lds.template load<16>(sb + a_off0); auto s1 = lds.template load<16>(sb + a_off1);
                    U32x8 t; __builtin_memcpy(&t.w[0], &s0, 16); __builtin_memcpy(&t.w[4], &s1, 16); __builtin_memcpy(&va, &t, 32);
                }
                {
                    U32x8 t[4];
                    #pragma unroll
                    for (int nb = 0; nb < 4; ++nb) {
                        auto s0 = lds.template load<16>(sb + b_off0[nb]); auto s1 = lds.template load<16>(sb + b_off1[nb]);
                        __builtin_memcpy(&t[nb].w[0], &s0, 16); __builtin_memcpy(&t[nb].w[4], &s1, 16);
                    }
                    __builtin_memcpy(&vb, &t[0], 128);
                }
                s_waitcnt_lgkmcnt(number<0>{});
                accum = mma(va, vb, accum, 127, 127, number<0>{}, number<0>{});
            }
        }
        if (tid == 128) SM_STAMP(rec, 6);
        const auto u_p = sks::partial_view(N);
        static_ford(mma.y_shape_c(), [&](auto im, auto rept, auto item, auto jn) {
            const auto y = make_tuple(im, rept, item, jn);
            const int m = u_cm(y), n = col_base + u_cn(y);
            if (m < M && n < N) {                                                // only live rows are stored
                const int acc_idx = static_cast<int>(u_acc(y));
                
#if SM_NT_STORES
                __builtin_nontemporal_store(accum[acc_idx], &args.P[u_p(split, m, n)]);          // r15 Q2: non-temporal partial (P is read once by another XCD)
#else
                args.P[u_p(split, m, n)] = accum[acc_idx];                       // raw FP32 partial; no scale, no BF16
#endif

            }
        });
        s_waitcnt_vmcnt(number<0>{});
        if (tid == 128) SM_STAMP(rec, 3);
    }
    if (tid == 0) SM_STAMP(rec, 4);
    if (tid == 128) SM_STAMP(rec, 7);
}
} // namespace smg

extern "C" __global__ __launch_bounds__(256)
void sm_gemm_v1(SmGemmArgsV1 args) {
#if defined(__gfx950__)
    if (args.sk != static_cast<uint32_t>(sks::SK) || args.K != 128u * sks::KBLOCKS || args.M == 0u || args.M > 32u) return;   // WG-uniform refusal
    __shared__ __align__(16) unsigned char stage[sks::LDS_BYTES];
    if (opus::block_id_x() == 0 && opus::block_id_y() == 0 && opus::thread_id_x() == 0) {   // split epoch words: GEMM reads ep_r (RAN-written), writes ep_g = ep_r + 1; RAN reads ep_g
        const uint64_t epr = __hip_atomic_load(reinterpret_cast<uint64_t*>(mori::cco::ccoGetLocalPtr(args.Cwin, SM_C_EPOCH)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint64_t*>(mori::cco::ccoGetLocalPtr(args.Cwin, SM_C_EPOCH_G)), epr + 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (args.M <= 16u) smg::gemm_body<1>(args, stage);
    else               smg::gemm_body<2>(args, stage);
#else
    __builtin_trap();
#endif
}
#endif // __HIP_DEVICE_COMPILE__

#ifndef __HIP_DEVICE_COMPILE__
extern "C" __global__ void sm_gemm_v1(SmGemmArgsV1) {}
#endif
