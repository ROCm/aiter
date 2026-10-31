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
#include "sm_abi_v3.h"
#include "stamps.hpp"
#include <opus/hip_minimal.hpp>

#ifdef __HIP_DEVICE_COMPILE__
#include <opus/opus.hpp>
#include "layout/sk_traits.hpp"
using namespace opus;

namespace smg {
// ---- SM_P_NT (R14-r22-stamps-analysis C1): P partials as 16 B non-temporal vector stores (no dirty P left in L2 => no end-of-kernel writeback of P). Source-validated layout facts (opus.hpp): the MFMA C tile is
// C:[(rept_c<y>, grpm_c<p>, pack_c<y>), (grpn_c<p>)] with pack_c = 4: a lane owns 4 CONSECUTIVE ROWS of one column (accumulator index stride 1 in `item`). A 4x4 transpose across the 4 adjacent lanes of a quad (DPP quad_perm 0x4E, 0xB1)
// turns that into 4 consecutive COLUMNS of one row = 16 B contiguous; the reduction order / MFMA chain is untouched, only data movement => P stays byte-identical. 0 = the scalar store of r21/r22.
OPUS_D float dpp_x1(float v) { return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xB1, 0xF, 0xF, true)); }   // quad_perm [1,0,3,2]: value of lane j^1
OPUS_D float dpp_x2(float v) { return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4E, 0xF, 0xF, true)); }   // quad_perm [2,3,0,1]: value of lane j^2
// in: a[k] = element k (row 4g+k) of THIS lane (column 4c+j, j = lane & 3); out: o[i] = element j of lane i of the quad = (row 4g+j, column 4c+i)
OPUS_D void quad_transpose4(const float (&a)[4], float (&o)[4], int j) {
    const bool b1 = (j & 2) != 0, b0 = (j & 1) != 0;
    const float p0 = dpp_x2(a[0]), p1 = dpp_x2(a[1]), p2 = dpp_x2(a[2]), p3 = dpp_x2(a[3]);
    const float n0 = b1 ? p2 : a[0], n1 = b1 ? p3 : a[1], n2 = b1 ? a[2] : p0, n3 = b1 ? a[3] : p1;
    const float q0 = dpp_x1(n0), q1 = dpp_x1(n1), q2 = dpp_x1(n2), q3 = dpp_x1(n3);
    o[0] = b0 ? q1 : n0; o[1] = b0 ? n1 : q0; o[2] = b0 ? q3 : n2; o[3] = b0 ? n3 : q2;
}
#if SM_P_NT
typedef float p_f4 __attribute__((ext_vector_type(4)));
// 16 B per lane. SM_P_NT = 1: `nt` (streaming) hint via the builtin. SM_P_NT = 2: inline `global_store_dwordx4 ... sc0 sc1 nt` (system-scope write-through + streaming; completion is awaited by the epilogue vmcnt(0) the code already has).
OPUS_D void st_p16(float* dst, const float (&o)[4]) {
    p_f4 w = {o[0], o[1], o[2], o[3]};
#if SM_P_NT == 2
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt" :: "v"(dst), "v"(w) : "memory");
#else
    __builtin_nontemporal_store(w, reinterpret_cast<p_f4*>(dst));
#endif
}
#endif
OPUS_D void bar() { asm volatile("" ::: "memory"); __builtin_amdgcn_s_barrier(); asm volatile("" ::: "memory"); }
// vmcnt(newer * MB), newer in [0, PFK-2] (r20: PFK 4 -> newer <= 2; the default case is the 4 * MB maximum)
// vmcnt immediate (6 bits, max 63): only newer in [0, PFK-2] is ever executed; a constant above 63 is clamped so the dead switch arms still compile (the static_assert in wait_newer proves the live ones fit). Fix for r22 BN128 (MB = 8 + AQ up to 16: the dead 4 * MB arm was 64).
template<int MB, int K> OPUS_D void wn() { if constexpr (K * MB <= 63) s_waitcnt_vmcnt(number<K * MB>{}); else s_waitcnt_vmcnt(number<63>{}); }
template<int MB> OPUS_D void wait_newer(int newer) {
    static_assert((sks::PFK - 2) * MB <= 63, "live vmcnt immediates must fit 6 bits: (PFK-2) * (BQ + AQ) <= 63");
    switch (newer) {
        case 0: s_waitcnt_vmcnt(number<0>{}); break;
        case 1: wn<MB, 1>(); break;
        case 2: wn<MB, 2>(); break;
        case 3: wn<MB, 3>(); break;
        default: wn<MB, 4>(); break;
    }
}
static_assert(sks::PFK >= 3 && sks::PFK - 2 <= 4, "wait_newer handles newer in [0, 4]");

struct U32x8 { uint32_t w[8]; };     // 32 fp8 = one MFMA A or B operand (no HIP vector types under RTC)

template<int AQ>
OPUS_D void gemm_body(const SmGemmArgsV1& args, unsigned char* stage) {
    constexpr int BQ = sks::B_INSTR_PER_WAVE;          // B vmem instructions per producer wave per K block: 4 (BN64) / 8 (BN128)
    constexpr int MB = BQ + AQ;                         // vmem instructions per producer wave per K block (AQ = ceil(M/16) <= 4 at BM64)
    constexpr int MT = (AQ + 1) / 2;                   // 32-row accumulator tiles per consumer wave that can hold live rows
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
    typename MMA::vtype_c accum[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) accum[t] = typename MMA::vtype_c{0};

    auto ga = make_gmem(static_cast<const fp8_t*>(args.A),
        static_cast<unsigned int>(static_cast<unsigned long long>(M - 1) * args.strideA + K));
    auto gb = make_gmem(static_cast<const fp8_t*>(args.W), static_cast<unsigned int>(static_cast<unsigned long long>(N) * K));
    const auto u_wl = sks::w_lane_view();                  // (half, lane, byte)
    const auto u_al = sks::a_lane_view(strideA);           // (row8, chunk, byte)

    if (producer) {
        // ---- per-lane source offsets (kb-independent parts) ----
        const int n16_0 = col_base / 16;
        // B: instruction q in [0,4): flat f = wave*4 + q -> band nb = f/2, half = f%2
        int b_src[BQ], b_dst[BQ];
        #pragma unroll
        for (int q = 0; q < BQ; ++q) {
            const int f = wave * BQ + q, nb = f >> 1, half = f & 1;
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
            for (int q = 0; q < BQ; ++q)
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
        const int cw = wave - sks::PRODUCER_WAVES;                                // 0: rows 0-15 of every 32-row tile, 1: rows 16-31
        const auto wc = sks::consumer_wave_coord(cw);
        const auto lc = sks::lane_coord(base.p_shape_c(), lane);
        const auto pc = opus::make_tuple(opus::get<0>(wc), opus::get<0>(lc), opus::get<1>(wc), opus::get<1>(lc));
        const auto u_cm = partition_layout_c(mma, opus::make_tuple(1_I, 0_I), pc);
        const auto u_cn = partition_layout_c(mma, opus::make_tuple(0_I, 1_I), pc);
        const auto u_acc = sks::accumulator_register_view(mma);
        // LDS fragment offsets (bytes inside a stage). A/B lane decode: l16 = lane%16 (row / n), g = lane/16 (K group 0..3).
        // r20: tile t (rows t*32 + cw*16 .. +15) is LIVE iff its lowest 16-row group index 2*t + cw < AQ (AQ = ceil(M/16)); the r18 `live_wave` is the AQ <= 2 special case. A dead tile does no LDS read and no MFMA (its outputs are masked).
        const int l16 = lane & 15, g = lane >> 4;
        const auto u_af = sks::a_frag_view();                                     // (row, seg, g), BM rows
        const auto u_bf = sks::b_frag_view();                                     // (nb, seg, g/2, g%2, n16)
        int a_off0[MT], a_off1[MT];
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int a_row = t * 32 + cw * 16 + l16;
            a_off0[t] = static_cast<int>(u_af(a_row, 0, g)); a_off1[t] = static_cast<int>(u_af(a_row, 1, g));
        }
        int b_off0[sks::EN], b_off1[sks::EN];
        #pragma unroll
        for (int nb = 0; nb < sks::EN; ++nb) {
            b_off0[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 0, g >> 1, g & 1, l16));
            b_off1[nb] = sks::A_STAGE_BYTES + static_cast<int>(u_bf(nb, 1, g >> 1, g & 1, l16));
        }
        auto lds = smem<fp8_t>(stage);
        for (int i = 0; i < L; ++i) {
            bar();                                                                // R_i
            if (i == 0 && tid == 128) SM_STAMP(rec, 5);
            if (cw < AQ) {                                                        // tile 0 of this wave is live (2*0 + cw < AQ); higher tiles are tested below
                const int sb = (i % sks::PFK) * sks::STAGE_BYTES;
                typename MMA::vtype_a va[MT]; typename MMA::vtype_b vb;
                #pragma unroll
                for (int t = 0; t < MT; ++t) {
                    if (2 * t + cw < AQ) {
                        auto s0 = lds.template load<16>(sb + a_off0[t]); auto s1 = lds.template load<16>(sb + a_off1[t]);
                        U32x8 tt; __builtin_memcpy(&tt.w[0], &s0, 16); __builtin_memcpy(&tt.w[4], &s1, 16); __builtin_memcpy(&va[t], &tt, 32);
                    }
                }
                {
                    U32x8 tt[sks::EN];
                    #pragma unroll
                    for (int nb = 0; nb < sks::EN; ++nb) {
                        auto s0 = lds.template load<16>(sb + b_off0[nb]); auto s1 = lds.template load<16>(sb + b_off1[nb]);
                        __builtin_memcpy(&tt[nb].w[0], &s0, 16); __builtin_memcpy(&tt[nb].w[4], &s1, 16);
                    }
                    __builtin_memcpy(&vb, &tt[0], sizeof(vb));
                }
                s_waitcnt_lgkmcnt(number<0>{});
                #pragma unroll
                for (int t = 0; t < MT; ++t)
                    if (2 * t + cw < AQ) accum[t] = mma(va[t], vb, accum[t], 127, 127, number<0>{}, number<0>{});
            }
        }
        if (tid == 128) SM_STAMP(rec, 6);
        const auto u_p = sks::partial_view(N);
        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            static_ford(mma.y_shape_c(), [&](auto im, auto rept, auto item, auto jn) {
                const auto y = opus::make_tuple(im, rept, item, jn);
#if SM_P_NT
                if (static_cast<int>(item) == 0) {                                   // one 16 B vector per (tile, band): the pack of 4 rows of this lane
                    const int j = lane & 3;
                    const int m0 = t * 32 + u_cm(y), n0c = col_base + u_cn(y);                      // element k = 0: row m0, column n0c = 16*band + 4*c + j
                    const int acc0 = static_cast<int>(u_acc(y));
                    const float a4[4] = {accum[t][acc0 + 0], accum[t][acc0 + 1], accum[t][acc0 + 2], accum[t][acc0 + 3]};
                    float o4[4];
                    smg::quad_transpose4(a4, o4, j);                                  // all lanes execute (wave-wide DPP), the store is predicated
                    const int m = m0 + j, n = n0c - j;                                // row m0 + j, columns n .. n + 3 (16 B aligned: n multiple of 4)
                    if (m < M && n + 3 < N) smg::st_p16(&args.P[u_p(split, m, n)], o4);   // only live rows are stored
                }
#else
                const int m = t * 32 + u_cm(y), n = col_base + u_cn(y);
                if (m < M && n < N) {                                            // only live rows are stored
                    const int acc_idx = static_cast<int>(u_acc(y));
#if SM_NT_STORES
                    __builtin_nontemporal_store(accum[t][acc_idx], &args.P[u_p(split, m, n)]);          // r15 Q2: non-temporal partial
#else
                    args.P[u_p(split, m, n)] = accum[t][acc_idx];                // raw FP32 partial; no scale, no BF16
#endif
                }
#endif
            });
        }
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
    if (args.sk != static_cast<uint32_t>(sks::SK) || args.K != 128u * sks::KBLOCKS || args.M == 0u || args.M > SM_MP) return;   // WG-uniform refusal
    __shared__ __align__(16) unsigned char stage[sks::LDS_BYTES];
    if (opus::block_id_x() == 0 && opus::block_id_y() == 0 && opus::thread_id_x() == 0) {   // split epoch words: GEMM reads ep_r (RAN-written), writes ep_g = ep_r + 1; RAN reads ep_g
        const uint64_t epr = __hip_atomic_load(reinterpret_cast<uint64_t*>(mori::cco::ccoGetLocalPtr(args.Cwin, SM_C_EPOCH)), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(reinterpret_cast<uint64_t*>(mori::cco::ccoGetLocalPtr(args.Cwin, SM_C_EPOCH_G)), epr + 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    switch ((args.M + 15u) / 16u) {                                                       // WG-uniform: AQ = ceil(M/16) in 1..8 (M <= SM_MP = 128)
        case 1u: smg::gemm_body<1>(args, stage); break;
        case 2u: smg::gemm_body<2>(args, stage); break;
        case 3u: smg::gemm_body<3>(args, stage); break;
        case 4u: smg::gemm_body<4>(args, stage); break;
        case 5u: smg::gemm_body<5>(args, stage); break;
        case 6u: smg::gemm_body<6>(args, stage); break;
        case 7u: smg::gemm_body<7>(args, stage); break;
        default: smg::gemm_body<8>(args, stage); break;
    }
#else
    __builtin_trap();
#endif
}
#endif // __HIP_DEVICE_COMPILE__

#ifndef __HIP_DEVICE_COMPILE__
extern "C" __global__ void sm_gemm_v1(SmGemmArgsV1) {}
#endif
