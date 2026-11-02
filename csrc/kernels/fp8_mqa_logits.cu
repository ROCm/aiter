// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prefill-phase FP8 MQA indexer logits kernel for gfx950 (GLM-5.x / DeepSeek-V3.2 lightning indexer):
//
//   logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]   for n in [ks[m], ke[m]), -inf elsewhere
//
// K is already gathered into a contiguous [N, 128] fp8 buffer with an [N] fp32 scale (no block table).
// Same contract as aiter.ops.triton.attention.fp8_mqa_logits.
//
// Design ("v35wes"):
//   * workgroup = 8 waves x 4 rows = 32 query rows; Q and w live in VGPRs, all waves walk the SAME K sequence,
//     so K is read from L2/HBM once per 32 rows.
//   * K stage = 2 x 32 tokens x 128 B staged in LDS (2-stage ring, bank-conflict-free swizzle), kv_scale staged
//     next to it; raw buffer-load prefetch 4 stages ahead with hand-placed s_waitcnt vmcnt (vmcnt also counts the
//     logits stores, so compiler-derived waits would drain them every stage).
//   * plain v_mfma_f32_32x32x64_f8f6f4 (2 per row per 32-token tile); SCALAR fp32 epilogue (v_max relu, v_fma
//     with w) interleaved one half-row per MFMA: on gfx950 scalar VALU co-issues under an independent MFMA while
//     packed v_pk_* VALU never does. Even/odd accumulation chains keep the summation order fixed.
//   * head reduction with v_permlane32_swap (inline asm: the clang builtin miscompiles its 2nd result on
//     ROCm 7.2.3), kv_scale applied once after the reduction, raw buffer stores; the add/mul/store tail of each
//     row pair is deferred by one sub-tile so it fills the MFMA->VALU hazard window (no s_nop in the loop).
//   * one s_barrier per stage, placed early (right after the last ds_read of the current stage) plus
//     a per-stage s_setprio staircase, so the two waves of a SIMD reach the barrier together instead of the older
//     wave idling there ~1/3 of every stage.
//   * per-WG stage range split into CLEAN stages (all 32 rows valid on every token: no compare/select) and MASKED
//     boundary stages; both ranges are shared across grid.y = SplitN workgroups. -inf outside [ks, ke) is written
//     by the kernel itself when clean_logits (no separate torch.full pass).
//   * host launcher picks (SplitN, reverse_rows) by a makespan model of the WG list (cost per WG / masked stage /
//     clean stage fitted on MI355X, round-robin over 8 XCDs). The model assumes the single-request causal layout
//     of a prefill chunk; multi-request chunks can do better with an explicit SplitN.
//
// Build flags (aiter/jit/optCompilerConfig.json, module_fp8_mqa_logits):
//   -fno-honor-nans            relu = single v_max (no IEEE canonicalize)
//   -fno-slp-vectorize         keep the epilogue scalar (SLP re-packs the fma pairs into v_pk_fma)
//   -mllvm -disable-machine-sink  otherwise the last row's MFMA pair of a stage sinks past the loop exit branch
#include "fp8_mqa_logits.h"
#include <hip/hip_fp8.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <type_traits>
template <int I> using IC = std::integral_constant<int, I>;

using fp8 = __hip_fp8_storage_t;
constexpr int WARP_SIZE = 64;
constexpr int NUM_HEADS = 32;
constexpr int HEAD_SIZE = 128;
#define CDIV_(a, b) (((a) + (b) - 1) / (b))
#define BUF_RSRC_WORD3 0x00020000
#define SYNC_TILE() __syncthreads()

namespace mqa_v35wes {

using f8x16 = __attribute__((__vector_size__(16))) fp8;
using f8x32 = __attribute__((__vector_size__(32))) fp8;
using f32x4 = __attribute__((__vector_size__(16))) float;
using f32x2 = __attribute__((__vector_size__(8))) float;
using u32x4 = __attribute__((__vector_size__(16))) unsigned int;
using i32x4 = __attribute__((__vector_size__(16))) int;
template <int N> __device__ __forceinline__ void wait_vmcnt() { static_assert(N >= 0 && N < 64, "vmcnt"); asm volatile("s_waitcnt vmcnt(%0)" :: "n"(N) : "memory"); }

// NOPS = s_nop wait states issued before the permlane: the caller guarantees >= 2 - NOPS independent instructions
// since the last VALU write of a / b (VALU write -> v_permlane32_swap read hazard; the compiler does not check it for
// inline asm, it only places the permlane -> VALU read wait after). __builtin_amdgcn_permlane32_swap is unusable
// on ROCm 7.2.3 clang: its second result is miscompiled to a copy of the first (port_1009/pl.hip).
template <int NOPS>
__device__ __forceinline__ void swap32_reduce2(float& a, float& b) {
    if constexpr (NOPS == 0) asm("v_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
    else if constexpr (NOPS == 1) asm("s_nop 0\n\tv_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
    else asm("s_nop 1\n\tv_permlane32_swap_b32 %0, %1" : "+v"(a), "+v"(b));
}
__device__ __forceinline__ f8x32 pack(f8x16 lo, f8x16 hi) {
    f8x32 v; *reinterpret_cast<f8x16*>(&v) = lo; *(reinterpret_cast<f8x16*>(&v) + 1) = hi; return v;
}

template <int NUM_WARPS, int SUB, int PF, bool CLEAN_LOGITS, bool REVERSE_ROWS>
__global__ __launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(const fp8* __restrict__ Q_ptr, const fp8* __restrict__ K_ptr, const float* __restrict__ kv_scale_ptr,
            const float* __restrict__ W_ptr, const int* __restrict__ ks_ptr, const int* __restrict__ ke_ptr,
            float* __restrict__ logits_ptr, int M, int N, int SplitN)
{
    constexpr int BLOCK_M    = 4;
    constexpr int ROWS_WG    = BLOCK_M * NUM_WARPS;
    constexpr int TILE_N     = 32;
    constexpr int TILE_BYTES = TILE_N * HEAD_SIZE;          // 4096 per sub-tile
    constexpr int STAGE_TOK  = TILE_N * SUB;
    constexpr int STAGE_BYTES= TILE_BYTES * SUB;
    constexpr int CHUNKS     = STAGE_BYTES / 16;            // 16-B chunks per stage
    constexpr int NTHREADS   = NUM_WARPS * WARP_SIZE;
    constexpr int STAGES     = 2;
    constexpr int NACC       = 16;
    constexpr int UNIT_SCALE = 0x7f;
    constexpr int A_SCALE    = 0x7f - 25;     // E8M0 2^-25 applied to the A operand (exact)
    constexpr float RESCALE  = 33554432.0f;   // 2^25
    using VecOutMFMA = __attribute__((__vector_size__(NACC * sizeof(float)))) float;

    __shared__ __attribute__((aligned(16))) fp8   k_lds[STAGES][STAGE_BYTES];
    __shared__ __attribute__((aligned(16))) float sc_lds[STAGES][STAGE_TOK];
    constexpr int SC_WAVES = STAGE_TOK / WARP_SIZE;          // waves that load the stage's scales (1 lane = 1 token)
    static_assert(STAGE_TOK % WARP_SIZE == 0 && SC_WAVES <= NUM_WARPS, "scale producer mapping");

    const int tid  = threadIdx.x;
    const int l    = tid % WARP_SIZE;
    const int warp = __builtin_amdgcn_readfirstlane(tid / WARP_SIZE);
    const int bx   = REVERSE_ROWS ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const int m0wg = bx * ROWS_WG;
    const int m0   = m0wg + warp * BLOCK_M;
    const int pid_split = blockIdx.y;
    if (m0wg >= M) return;

    const int rw    = l / 16;
    const int col   = l % 16;
    const int nA    = col + 16 * (rw & 1);   // token within tile for the B operand (0..31)
    const int dhalf = rw / 2;                // which 16-B chunk pair of each 64-dim half
    const int jtok  = l % 32;                // output token (after permlane reduction)
    const int hset  = l / 32;                // 0: row r, 1: row r+1

    // ---- my rows: Q (A operand layout) and W to registers ----
    f8x32 q_reg[BLOCK_M][2];
    f32x2 w2[BLOCK_M][NACC / 2];   // native 64-bit pairs (asm operands need aligned pairs)
    int   ks_reg[BLOCK_M], ke_reg[BLOCK_M];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; ++r) {
        const int m = m0 + r;
        if (m >= M) { ks_reg[r] = 0; ke_reg[r] = 0;
            #pragma unroll
            for (int v = 0; v < NACC / 2; ++v) w2[r][v] = f32x2{0.f, 0.f};
            q_reg[r][0] = q_reg[r][1] = f8x32{}; continue; }
        ks_reg[r] = ks_ptr[m]; ke_reg[r] = ke_ptr[m];
        const int64_t qb = ((int64_t)m * NUM_HEADS + nA) * HEAD_SIZE;
        q_reg[r][0] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + dhalf * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (2 + dhalf) * 16]));
        q_reg[r][1] = pack(*reinterpret_cast<const f8x16*>(&Q_ptr[qb + (4 + dhalf) * 16]),
                           *reinterpret_cast<const f8x16*>(&Q_ptr[qb + (6 + dhalf) * 16]));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const f32x4 wv = *reinterpret_cast<const f32x4*>(&W_ptr[(int64_t)m * NUM_HEADS + 8 * j + 4 * hset]);
            w2[r][2 * j]     = f32x2{wv[0], wv[1]};   // v35v: no prescale (plain MFMA)
            w2[r][2 * j + 1] = f32x2{wv[2], wv[3]};
        }
    }
    // ---- WG-wide tile range (uniform across waves) ----
    int lo = INT_MAX, hi = 0, max_ks = 0, min_ke = INT_MAX;
    for (int r = 0; r < ROWS_WG; ++r) {
        const int m = m0wg + r;
        if (m < M) { const int a = ks_ptr[m], b = ke_ptr[m]; lo = min(lo, a); hi = max(hi, b); max_ks = max(max_ks, a); min_ke = min(min_ke, b); }
    }
    lo = __builtin_amdgcn_readfirstlane(max(lo, 0)); hi = __builtin_amdgcn_readfirstlane(min(hi, N));
    max_ks = __builtin_amdgcn_readfirstlane(max_ks); min_ke = __builtin_amdgcn_readfirstlane(min(min_ke, N));
    const bool empty = (lo >= hi);
    lo = empty ? 0 : (lo & ~(TILE_N - 1));
    const int num_tiles   = empty ? 0 : CDIV_(hi - lo, STAGE_TOK);   // stages
    const int tile_end    = min(N, lo + num_tiles * STAGE_TOK);
    // clean stage range [tc0, tc1): every token of the stage is inside [ks,ke) of ALL rows of the WG, all rows exist, < N
    const bool all_rows = (m0wg + ROWS_WG) <= M;
    int tc0 = min(CDIV_(max(max_ks - lo, 0), STAGE_TOK), num_tiles);
    int tc1 = all_rows ? min(max((min_ke - lo) / STAGE_TOK, 0), num_tiles) : 0;
    tc1 = max(tc1, tc0);
    const int n_masked = tc0 + (num_tiles - tc1);               // stages [0,tc0) U [tc1,num_tiles), contiguous index space
    const float NEG_INF   = -__builtin_huge_valf();

    if constexpr (CLEAN_LOGITS) {
        if (pid_split == 0) {   // -inf for [0, lo) and [tile_end, N) of my rows (wave-local, 64 lanes)
            const f32x4 vneg = {NEG_INF, NEG_INF, NEG_INF, NEG_INF};
            auto fill_range = [&](float* row, int64_t row_base, int a, int b) {
                if (((row_base + a) & 3) == 0) {
                    const int vb = a + ((b - a) & ~3);
                    for (int i = a + l * 4; i < vb; i += WARP_SIZE * 4) *reinterpret_cast<f32x4*>(row + i) = vneg;
                    for (int i = vb + l; i < b; i += WARP_SIZE) row[i] = NEG_INF;
                } else {
                    for (int i = a + l; i < b; i += WARP_SIZE) row[i] = NEG_INF;
                }
            };
            #pragma unroll 1
            for (int r = 0; r < BLOCK_M; ++r) {
                const int m = m0 + r;
                if (m >= M) break;
                const int64_t row_base = (int64_t)m * N;
                fill_range(logits_ptr + row_base, row_base, 0, min(lo, N));
                fill_range(logits_ptr + row_base, row_base, tile_end, N);
            }
        }
    }
    if (empty) return;

    // ---- buffer resources ----
    const __amdgpu_buffer_rsrc_t k_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)K_ptr, 0, N * HEAD_SIZE, BUF_RSRC_WORD3);
    const __amdgpu_buffer_rsrc_t s_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)kv_scale_ptr, 0, N * 4, BUF_RSRC_WORD3);
    (void)s_rsrc;
    // raw V# for the inline-asm prefetch loads (same fields as make_buffer_rsrc: stride 0, raw buffer)
    auto raw_rsrc = [&](const void* p, int bytes) {
        const uint64_t a = reinterpret_cast<uint64_t>(p);
        i32x4 r;
        r[0] = __builtin_amdgcn_readfirstlane((int)(uint32_t)a);
        r[1] = __builtin_amdgcn_readfirstlane((int)(uint32_t)(a >> 32) & 0xffff);
        r[2] = __builtin_amdgcn_readfirstlane(bytes);
        r[3] = BUF_RSRC_WORD3;
        return r;
    };
    const i32x4 k_v = raw_rsrc(K_ptr, N * HEAD_SIZE);
    const i32x4 s_v = raw_rsrc(kv_scale_ptr, N * 4);
    __amdgpu_buffer_rsrc_t o_rsrc[BLOCK_M / 2];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; r += 2) {
        const int row = __builtin_amdgcn_readfirstlane(min(m0 + r, M - 1));
        o_rsrc[r / 2] = __builtin_amdgcn_make_buffer_rsrc((void*)(logits_ptr + (int64_t)row * N), 0, 2 * N * 4, BUF_RSRC_WORD3);
    }
    const int o_voff = hset * N * 4 + jtok * 4;

    // ---- producer mapping: chunk c (16 B) of the tile: token c/8, 16-B piece c%8 ----
    // Each lane copies CPL chunks; the LDS destination XOR-swizzles the piece index by (token & 7).
    constexpr int CPL = CDIV_(CHUNKS, NTHREADS);
    static_assert(CHUNKS % NTHREADS == 0, "every lane issues exactly CPL K loads per stage (vmcnt accounting)");
    // vmcnt accounting (per wave, lower bound: K loads + logits stores; wave 0's scale loads only add younger ops)
    constexpr int ST_PER_STAGE = 2 * SUB;                       // pair_store per stage
    constexpr int OPS_STAGE    = CPL + ST_PER_STAGE;
    constexpr int W_PRO        = (PF - 1) * CPL;                 // prologue: slot 0, younger = slots 1..PF-1
    constexpr int W_STEADY     = ST_PER_STAGE + (PF - 1) * OPS_STAGE;
    static_assert(W_STEADY < 64, "vmcnt field is 6 bits");
    int   pf_lds_off[CPL];
    int   pf_voff[CPL];
    bool  pf_on[CPL];
    #pragma unroll
    for (int i = 0; i < CPL; ++i) {
        const int c = tid + i * NTHREADS;
        pf_on[i] = c < CHUNKS;
        const int t = c / 8, p = c % 8;             // t = token within stage (0..STAGE_TOK-1)
        pf_voff[i]    = c * 16;
        pf_lds_off[i] = t * HEAD_SIZE + ((p ^ ((t >> 1) & 7)) * 16);
    }
    // per-lane LDS read offsets for the B operand (4 x 16 B): pieces dhalf, 2+dhalf, 4+dhalf, 6+dhalf of token nA
    const int rd_base = nA * HEAD_SIZE;
    const int sw = (nA >> 1) & 7;
    const int rd0 = rd_base + (((0 + dhalf) ^ sw) * 16);
    const int rd1 = rd_base + (((2 + dhalf) ^ sw) * 16);
    const int rd2 = rd_base + (((4 + dhalf) ^ sw) * 16);
    const int rd3 = rd_base + (((6 + dhalf) ^ sw) * 16);

    struct Bop { f8x32 b0, b1; float scale; };
    auto read_lds = [&](Bop& B, auto stage_c, auto sub_c) {
        constexpr int stage = decltype(stage_c)::value, sub = decltype(sub_c)::value;
        constexpr int so = sub * TILE_BYTES;
        B.b0 = pack(*reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd0]), *reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd1]));
        B.b1 = pack(*reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd2]), *reinterpret_cast<const f8x16*>(&k_lds[stage][so + rd3]));
        B.scale = sc_lds[stage][sub * TILE_N + jtok];   // needed only at pair_store, long after the MFMA's lgkmcnt wait
    };
    auto mfma_row = [&](int r, const Bop& B, VecOutMFMA& c) {
        VecOutMFMA z = {};
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][0], B.b0, z, 0, 0, 0, 0, 0, 0);
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][1], B.b1, c, 0, 0, 0, 0, 0, 0);
    };
    auto mfma_a = [&](int r, const Bop& B, VecOutMFMA& c) {
        VecOutMFMA z = {};
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][0], B.b0, z, 0, 0, 0, 0, 0, 0);
    };
    auto mfma_b = [&](int r, const Bop& B, VecOutMFMA& c) {
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][1], B.b1, c, 0, 0, 0, 0, 0, 0);
    };
    // half H of row r: elements 8H..8H+7; even/odd chains == lanes x/y of the v35 packed epilogue (same rounding order)
    auto epi_half = [&](int r, const VecOutMFMA& vC, auto h_c, float& ae, float& ao) {
        constexpr int H_ = decltype(h_c)::value;
        #pragma unroll
        for (int v = 4 * H_; v < 4 * H_ + 4; ++v) {
            const float ye = __builtin_fmaxf(vC[2 * v], 0.f), yo = __builtin_fmaxf(vC[2 * v + 1], 0.f);
            if (v == 0) { ae = ye * w2[r][0][0]; ao = yo * w2[r][0][1]; }
            else { ae = __builtin_fmaf(ye, w2[r][v][0], ae); ao = __builtin_fmaf(yo, w2[r][v][1], ao); }
        }
    };
    // per-lane window bounds of "my" row in the pair (lanes 0-31: row r, 32-63: row r+1)
    int sel_ks[BLOCK_M / 2], sel_ke[BLOCK_M / 2], sel_ok[BLOCK_M / 2];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; r += 2) {
        sel_ks[r / 2] = hset ? ks_reg[r + 1] : ks_reg[r];
        sel_ke[r / 2] = hset ? ke_reg[r + 1] : ke_reg[r];
        sel_ok[r / 2] = (m0 + r + hset < M);
    }
    const unsigned OOB_VOFF = 0x7ffffff0u;   // beyond num_records -> store dropped by the range check
    // pair_finish: everything after the permlane (p0 / p1 already swapped); pair_store = swap + finish
    auto pair_finish = [&](int r, float p0, float p1, float scale, int token_base, auto masked_c) {
        constexpr bool MASKED = decltype(masked_c)::value;
        const float res = (p0 + p1) * scale;
        if constexpr (!MASKED) {
            __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, res), o_rsrc[r / 2], (unsigned)o_voff, token_base * 4, 0);
        } else {
            const int abs_pos = token_base + jtok;
            const bool valid  = abs_pos >= sel_ks[r / 2] && abs_pos < sel_ke[r / 2];
            const bool write  = sel_ok[r / 2] && abs_pos < N;
            float val;
            if constexpr (CLEAN_LOGITS) val = valid ? res : NEG_INF;
            else { val = res; }
            const bool do_write = CLEAN_LOGITS ? write : (write && valid);
            const unsigned voff = do_write ? (unsigned)o_voff : OOB_VOFF;
            __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, val), o_rsrc[r / 2], voff, token_base * 4, 0);
        }
    };
    auto pair_store = [&](int r, float p0, float p1, float scale, int token_base, auto masked_c, auto nops_c) {
        swap32_reduce2<decltype(nops_c)::value>(p0, p1);
        pair_finish(r, p0, p1, scale, token_base, masked_c);
    };

    // ---- one pipeline over a contiguous stage-index range [tb, te); MASKED maps the index onto [0,tc0) U [tc1,num_tiles) ----
    auto run_pipeline = [&](auto masked_c, const int tb, const int te) {
        constexpr bool MASKED = decltype(masked_c)::value;
        if (tb >= te) return;
        auto tokbase = [&](int t) -> int {
            int st = t;
            if constexpr (MASKED) st = (t < tc0) ? t : (t - tc0 + tc1);
            return lo + st * STAGE_TOK;
        };
    u32x4 pf_reg0[CPL], pf_reg1[CPL], pf_reg2[CPL], pf_reg3[CPL];
    float pf_sc0 = 0.f, pf_sc1 = 0.f, pf_sc2 = 0.f, pf_sc3 = 0.f;   // scale of token tid of the stage (waves < SC_WAVES)

    auto issue_loads = [&](int t, auto slot_c) {
        constexpr int slot = decltype(slot_c)::value;
        u32x4* pf = slot == 0 ? pf_reg0 : slot == 1 ? pf_reg1 : slot == 2 ? pf_reg2 : pf_reg3;
        float& ps = slot == 0 ? pf_sc0 : slot == 1 ? pf_sc1 : slot == 2 ? pf_sc2 : pf_sc3;
        const int token_base = __builtin_amdgcn_readfirstlane(tokbase(t));
        const int soff = token_base * HEAD_SIZE;
        #pragma unroll
        for (int i = 0; i < CPL; ++i)
            asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen" : "=v"(pf[i]) : "v"(pf_voff[i]), "s"(k_v), "s"(soff) : "memory");
        if (warp < SC_WAVES)   // wave-uniform (SGPR) branch; extra younger ops only make wave 0's waits stricter
        {
            const int sc_voff = tid * 4, sc_soff = token_base * 4;
            const i32x4 sv = s_v;
            asm volatile("buffer_load_dword %0, %1, %2, %3 offen" : "=v"(ps) : "v"(sc_voff), "s"(sv), "s"(sc_soff) : "memory");
        }
    };
    auto write_lds = [&](auto slot_c, auto stage_c, auto wait_c) {
        constexpr int slot = decltype(slot_c)::value, stage = decltype(stage_c)::value;
        wait_vmcnt<decltype(wait_c)::value>();
        const u32x4* pf = slot == 0 ? pf_reg0 : slot == 1 ? pf_reg1 : slot == 2 ? pf_reg2 : pf_reg3;
        const float ps = slot == 0 ? pf_sc0 : slot == 1 ? pf_sc1 : slot == 2 ? pf_sc2 : pf_sc3;
        #pragma unroll
        for (int i = 0; i < CPL; ++i)
            *reinterpret_cast<u32x4*>(&k_lds[stage][pf_lds_off[i]]) = pf[i];
        if (warp < SC_WAVES) sc_lds[stage][tid] = ps;
    };

    // ---- pipeline ----
    // pf slot s: tile regs (global -> LDS staging); LDS stage: 2-ring; Bop ring: operands of tile t (cur) and t+1 (nxt)
    Bop Bo[2];
    Bo[1].scale = 0.f;
    VecOutMFMA acc[2];
    // previous sub-tile, stored during the next one: P0/P1 = rows 0/1 after the permlane, p2 and (e3, o3) of rows 2/3
    float pd_P0 = 0.f, pd_P1 = 0.f, pd_p2 = 0.f, pd_e3 = 0.f, pd_o3 = 0.f;
    int pd_tok = 0;
    auto dummy_store = [&]() { __builtin_amdgcn_raw_buffer_store_b32(0u, o_rsrc[0], OOB_VOFF, 0, 0); };
#define FLUSH_PEND() { pair_finish(0, pd_P0, pd_P1, Bo[1].scale, pd_tok, MK{}); \
                       pair_store(2, pd_p2, pd_e3 + pd_o3, Bo[1].scale, pd_tok, MK{}, IC<2>{}); }
    // prologue: stages t..t+PF-1 in register slots; stage t -> LDS stage 0; row 0 of sub-tile 0 -> acc[0]
    issue_loads(tb, IC<0>{});
    if constexpr (PF > 1) issue_loads(min(tb + 1, te - 1), IC<1>{});
    if constexpr (PF > 2) issue_loads(min(tb + 2, te - 1), IC<2>{});
    if constexpr (PF > 3) issue_loads(min(tb + 3, te - 1), IC<3>{});
    write_lds(IC<0>{}, IC<0>{}, IC<W_PRO>{});
    issue_loads(min(tb + PF, te - 1), IC<0>{});
    __syncthreads();
    read_lds(Bo[0], IC<0>{}, IC<0>{});
    mfma_a(0, Bo[0], acc[0]); mfma_b(0, Bo[0], acc[0]);

    // sub-tile compute: operands in Bo[BC]; the NEXT sub-tile's operands are fetched into Bo[BN] mid-way
    // (RD_NEXT = the read_lds call for the next sub-tile; issued between mfma_row(3) and the tail rows).
// Wait-state bookkeeping. A 32x32x64 XDL result needs 20 instructions before a VALU reads it; each row's epilogue
// starts right after an "a" MFMA, with only 16 VALU + 1 MFMA since the producing "b" MFMA -> 2 COMPILER-VISIBLE
// fillers are needed (the compiler counts an inline-asm block as 0 wait states, so the permlane asm cannot fill).
// The e/o adds, the add/mul/store tails and the permlanes of row pairs are deferred to fill those slots:
//   a1 | p3' = e3' + o3', finish(P0', P1') | epi row 0 h0      b1 | permlane(p2', p3') | epi row 0 h1
//   a2 | p0 = e0 + o0, finish(p2', p3')    | epi row 1 h0      b2 | epi row 1 h1
//   a3 | RD_NEXT ds_reads                  | epi row 2 h0      b3 | epi row 2 h1
//   a0+| p1 = e1 + o1, p2 = e2 + o2         | epi row 3 h0      b0+| permlane(p0, p1) | epi row 3 h1
// (' = previous sub-tile). Every permlane input was written >= 2 instructions earlier (VALU -> permlane hazard).
#define SUBTILE(BC, BN, TOKBASE, RD_NEXT, MK, PEND, PS, PM, RD_LATE) \
    { \
        float e0, o0, e1, o1, e2, o2; \
        if constexpr ((PS) >= 0) __builtin_amdgcn_s_setprio(PS); \
        mfma_a(1, Bo[BC], acc[1]); \
        __builtin_amdgcn_sched_barrier(0); \
        float pd_p3 = 0.f; \
        if constexpr (PEND) { pd_p3 = pd_e3 + pd_o3; pair_finish(0, pd_P0, pd_P1, Bo[BN].scale, pd_tok, MK{}); } \
        else dummy_store(); \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(0, acc[0], IC<0>{}, e0, o0); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_b(1, Bo[BC], acc[1]); \
        __builtin_amdgcn_sched_barrier(0); \
        if constexpr (PEND) swap32_reduce2<0>(pd_p2, pd_p3); \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(0, acc[0], IC<1>{}, e0, o0); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_a(2, Bo[BC], acc[0]); \
        __builtin_amdgcn_sched_barrier(0); \
        float p0 = e0 + o0; \
        if constexpr (PEND) pair_finish(2, pd_p2, pd_p3, Bo[BN].scale, pd_tok, MK{}); else dummy_store(); \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(1, acc[1], IC<0>{}, e1, o1); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_b(2, Bo[BC], acc[0]); \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(1, acc[1], IC<1>{}, e1, o1); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_a(3, Bo[BC], acc[1]); \
        __builtin_amdgcn_sched_barrier(0); \
        if constexpr ((PM) >= 0) __builtin_amdgcn_s_setprio(PM); \
        RD_NEXT; \
        epi_half(2, acc[0], IC<0>{}, e2, o2); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_b(3, Bo[BC], acc[1]); \
        __builtin_amdgcn_sched_barrier(0); \
        RD_LATE; \
        epi_half(2, acc[0], IC<1>{}, e2, o2); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_a(0, Bo[BN], acc[0]); \
        __builtin_amdgcn_sched_barrier(0); \
        float p1 = e1 + o1; \
        pd_p2 = e2 + o2; \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(3, acc[1], IC<0>{}, pd_e3, pd_o3); \
        __builtin_amdgcn_sched_barrier(0); \
        mfma_b(0, Bo[BN], acc[0]); \
        __builtin_amdgcn_sched_barrier(0); \
        swap32_reduce2<0>(p0, p1); \
        pd_P0 = p0; pd_P1 = p1; \
        __builtin_amdgcn_sched_barrier(0); \
        epi_half(3, acc[1], IC<1>{}, pd_e3, pd_o3); \
        pd_tok = TOKBASE; \
        __builtin_amdgcn_sched_barrier(0); \
    }
    // one LDS stage: SUB sub-tiles.
    // Early barrier: the stage barrier only has to follow (RAW) the stage-start ds_write of NXT and (WAR) the last
    // ds_read of CUR, so it sits after MFMA b(3) of the sub-tile that issues that last CUR read. The next-stage
    // operand read of the last sub-tile is then a plain ds_read after a(3), its latency covered by epi(2) + b(3)
    // like every other sub-tile. At pipeline exit only the dummy NXT read follows the last barrier (result unused).
    // Priority staircase: barrier-to-barrier = last sub-tile of a stage, then the first sub-tile of the next one;
    // every wave steps s_setprio 3 -> 2 -> 1 -> 0 along that span (PS at sub-tile start, PM after a(3)), so the
    // leading wave (normally the older one, which wins issue arbitration) yields the issue port to the lagging
    // wave and both reach the barrier together instead of one idling ~1/3 of the stage. -1 = no s_setprio.
#define NEXT_STAGE_READ(NXT) read_lds(Bo[0], IC<NXT>{}, IC<0>{})
#define STAGE_STEP(CUR, NXT, WS, TT, MK, WC, P0)                                                          \
    {                                                                                                     \
        const int stage_base = __builtin_amdgcn_readfirstlane(tokbase(TT));                      \
        write_lds(IC<WS>{}, IC<NXT>{}, IC<WC>{});                                                        \
        issue_loads(min((TT) + PF + 1, te - 1), IC<WS>{});                                             \
        if constexpr (SUB == 2) {                                                                         \
            SUBTILE(0, 1, stage_base, read_lds(Bo[1], IC<CUR>{}, IC<1>{}), MK, P0, 1, 0, SYNC_TILE())   \
            SUBTILE(1, 0, stage_base + TILE_N, NEXT_STAGE_READ(NXT), MK, true, 3, 2, (void)0) \
        } else {                                                                                          \
            SUBTILE(0, 1, stage_base, read_lds(Bo[1], IC<CUR>{}, IC<1>{}), MK, P0, 1, -1, (void)0)   \
            SUBTILE(1, 0, stage_base + TILE_N, read_lds(Bo[0], IC<CUR>{}, IC<2>{}), MK, true, 0, -1, (void)0)   \
            SUBTILE(0, 1, stage_base + 2 * TILE_N, read_lds(Bo[1], IC<CUR>{}, IC<3>{}), MK, true, 3, -1, SYNC_TILE())   \
            SUBTILE(1, 0, stage_base + 3 * TILE_N, NEXT_STAGE_READ(NXT), MK, true, 2, -1, (void)0) \
        }                                                                                                 \
    }

    static_assert(SUB == 2 || SUB == 4, "v14 supports SUB = 2 or 4");
    static_assert(PF == 2 || PF == 4, "v14 supports PF = 2 or 4 register slots");
    using MK = std::remove_cv_t<std::remove_reference_t<decltype(masked_c)>>;
    // first iteration peeled: slot k was loaded in the prologue -> younger = (PF-k)*CPL + (k-1)*OPS_STAGE
    constexpr int W1 = (PF - 1) * CPL, W2 = (PF - 2) * CPL + OPS_STAGE;
    constexpr int W3 = (PF - 3) * CPL + 2 * OPS_STAGE, W4 = 3 * OPS_STAGE;
    if constexpr (PF == 4) {
        STAGE_STEP(0, 1, 1, tb, MK, W1, false)
        if (tb + 1 >= te) { FLUSH_PEND(); return; }
        STAGE_STEP(1, 0, 2, tb + 1, MK, W2, true)
        if (tb + 2 >= te) { FLUSH_PEND(); return; }
        STAGE_STEP(0, 1, 3, tb + 2, MK, W3, true)
        if (tb + 3 >= te) { FLUSH_PEND(); return; }
        STAGE_STEP(1, 0, 0, tb + 3, MK, W4, true)
    #pragma unroll 1
    for (int t = tb + 4; t < te; t += 4) {
        STAGE_STEP(0, 1, 1, t, MK, W_STEADY, true)
        if (t + 1 >= te) break;
        STAGE_STEP(1, 0, 2, t + 1, MK, W_STEADY, true)
        if (t + 2 >= te) break;
        STAGE_STEP(0, 1, 3, t + 2, MK, W_STEADY, true)
        if (t + 3 >= te) break;
        STAGE_STEP(1, 0, 0, t + 3, MK, W_STEADY, true)
    }
    FLUSH_PEND();
    } else {
        STAGE_STEP(0, 1, 1, tb, MK, W1, false)
        if (tb + 1 >= te) { FLUSH_PEND(); return; }
        STAGE_STEP(1, 0, 0, tb + 1, MK, OPS_STAGE, true)
    #pragma unroll 1
    for (int t = tb + 2; t < te; t += 2) {
        STAGE_STEP(0, 1, 1, t, MK, W_STEADY, true)
        if (t + 1 >= te) break;
        STAGE_STEP(1, 0, 0, t + 1, MK, W_STEADY, true)
    }
    FLUSH_PEND();
    }
#undef STAGE_STEP
#undef NEXT_STAGE_READ
#undef FLUSH_PEND
#undef SUBTILE
    };

    // ---- work split: masked stages and clean stages are each shared across the SplitN workgroups ----
    auto share = [&](int n, int& b_, int& e_) { const int per = CDIV_(n, SplitN); b_ = pid_split * per; e_ = min(n, b_ + per); };
    int mb, me, cb, ce;
    share(n_masked, mb, me);
    share(tc1 - tc0, cb, ce); cb += tc0; ce += tc0;
    run_pipeline(std::true_type{}, mb, me);
    wait_vmcnt<0>();   // no asm prefetch load may still be in flight when the next pipeline reuses its VGPRs
    run_pipeline(std::false_type{}, cb, ce);
    wait_vmcnt<0>();
}



}  // namespace mqa_v35wes

#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <vector>
namespace {
// Makespan model of the launch: WG cost = GA + GBM * masked stages + GBC * clean stages (us, fit on MI355X),
// WGs dispatched in blockIdx order round-robin over 8 XCDs, greedy onto the CUs of each XCD (1 WG per CU).
constexpr double GA = 9.87, GBM = 2.962, GBC = 1.975;
double model_makespan(int M, int N, int split, int rev, int cus_per_xcd) {
    const int rb = CDIV_(M, 32);
    std::vector<std::priority_queue<double, std::vector<double>, std::greater<double>>> xcd(8);
    for (auto& q : xcd) for (int i = 0; i < cus_per_xcd; ++i) q.push(0.0);
    double mk = 0.0; int idx = 0;
    for (int y = 0; y < split; ++y)
        for (int x = 0; x < rb; ++x, ++idx) {
            const int bx = rev ? rb - 1 - x : x;
            const int r0 = bx * 32, r1 = std::min(M, r0 + 32);
            auto ke = [&](int m) { return std::min((4 * N - M + m + 1) / 4, N); };
            const int hi = ke(r1 - 1), mn = ke(r0);
            const int n = CDIV_(hi, 64);
            const int tc1 = (r0 + 32 <= M) ? std::min(mn / 64, n) : 0;
            auto share = [&](int n_) { const int per = CDIV_(n_, split); const int b = y * per; return std::max(0, std::min(n_, b + per) - b); };
            auto& q = xcd[idx % 8];
            const double t = q.top() + GA + GBM * share(n - tc1) + GBC * share(tc1);
            q.pop(); q.push(t); mk = std::max(mk, t);
        }
    return mk;
}
void choose_grid(int M, int N, int total_cu, int& split, int& rev) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int>, std::pair<int, int>> cache;
    std::lock_guard<std::mutex> g(mu);
    const auto key = std::make_tuple(M, N, total_cu);
    auto it = cache.find(key);
    if (it != cache.end()) { split = it->second.first; rev = it->second.second; return; }
    const int cus_per_xcd = std::max(1, total_cu / 8);
    double best = 1e300; split = 1; rev = 1;
    for (int s = 1; s <= 16; ++s) for (int r = 0; r < 2; ++r) {
        const double t = model_makespan(M, N, s, r, cus_per_xcd);
        if (t < best * 0.999) { best = t; split = s; rev = r; }
    }
    cache[key] = {split, rev};
}

void dispatch_fp8_mqa_logits(const fp8* Q, const fp8* K, const float* scale, const float* w, const int* ks,
                             const int* ke, float* out, int M, int N, int split_n, int num_warps, bool clean,
                             bool rev, hipStream_t stream) {
    const int row_blocks = CDIV_(M, 4 * num_warps);
    dim3 grid(row_blocks, split_n);
#define L(NW, CL, RV) \
    mqa_v35wes::kernel<NW, 2, 4, CL, RV><<<grid, NW * WARP_SIZE, 0, stream>>>(Q, K, scale, w, ks, ke, out, M, N, split_n)
#define LC(NW) \
    if (clean) { if (rev) L(NW, true, true); else L(NW, true, false); } \
    else       { if (rev) L(NW, false, true); else L(NW, false, false); }
    if (num_warps == 8) { LC(8); } else { LC(4); }
#undef LC
#undef L
}
}  // namespace

// ============================================================================
// Torch-facing entry point
// ============================================================================
torch::Tensor fp8_mqa_logits(
    torch::Tensor q_fp8,        // [M, 32, 128] fp8
    torch::Tensor k_fp8,        // [N, 128]     fp8
    torch::Tensor kv_scale,     // [N]          fp32
    torch::Tensor weights,      // [M, 32]      fp32
    torch::Tensor cu_seqlen_ks, // [M]          int32
    torch::Tensor cu_seqlen_ke, // [M]          int32
    int64_t BlockM,
    int64_t SplitN,
    int64_t num_warps,
    int64_t TotalCuCount,
    bool clean_logits,
    int64_t Unroll2,
    int64_t ReverseRows,
    std::optional<torch::Tensor> out)
{
    const int M = q_fp8.size(0);
    const int N = k_fp8.size(0);

    TORCH_CHECK(q_fp8.dim() == 3 && q_fp8.size(1) == NUM_HEADS && q_fp8.size(2) == HEAD_SIZE,
                "Only n_heads=32, head_dim=128 supported");
    TORCH_CHECK(k_fp8.dim() == 2 && k_fp8.size(1) == HEAD_SIZE, "K must be [N, 128]");
    TORCH_CHECK(q_fp8.element_size() == 1 && k_fp8.element_size() == 1, "q_fp8 and k_fp8 must be fp8");
    TORCH_CHECK(q_fp8.is_contiguous() && k_fp8.is_contiguous(), "q_fp8 and k_fp8 must be contiguous");
    TORCH_CHECK(kv_scale.scalar_type() == torch::kFloat32 && kv_scale.is_contiguous() && kv_scale.numel() == N,
                "kv_scale must be a contiguous f32 [N] tensor");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32 && weights.is_contiguous() &&
                    weights.numel() == (int64_t)M * NUM_HEADS,
                "weights must be a contiguous f32 [M, 32] tensor");
    TORCH_CHECK(cu_seqlen_ks.scalar_type() == torch::kInt32 && cu_seqlen_ke.scalar_type() == torch::kInt32 &&
                    cu_seqlen_ks.is_contiguous() && cu_seqlen_ke.is_contiguous() &&
                    cu_seqlen_ks.numel() == M && cu_seqlen_ke.numel() == M,
                "cu_seqlen_ks/ke must be contiguous int32 [M] tensors");
    // Rows per wave are fixed at 4 (32 per workgroup); BlockM is kept for drop-in compatibility only.
    TORCH_CHECK(BlockM == 0 || BlockM == 4, "BlockM must be 0 (default) or 4, got ", BlockM);
    (void)Unroll2;   // no unroll variant: the stage loop is always unrolled x4 over the LDS/prefetch ring

    int num_warps_ = num_warps <= 0 ? 8 : static_cast<int>(num_warps);
    TORCH_CHECK(num_warps_ == 8 || num_warps_ == 4, "num_warps must be 0 (default), 4 or 8, got ", num_warps);
    int split_n = static_cast<int>(SplitN);
    int rev     = static_cast<int>(ReverseRows);
    if (split_n <= 0) {
        if (num_warps_ == 8) {
            int s, r;
            choose_grid(M, N, static_cast<int>(TotalCuCount), s, r);
            split_n = s;
            if (rev < 0) rev = r;
        } else {
            split_n = std::max(1, CDIV_(static_cast<int>(TotalCuCount) * 8 / num_warps_, std::max(1, CDIV_(M, 16))));
        }
    }
    if (rev < 0) rev = (N > 2048);

    // clean_logits: the kernel writes -inf outside every row's window itself, so the buffer can start
    // uninitialised. Without clean_logits positions outside the window are left untouched (only safe when the
    // consumer is range aware, e.g. top_k_per_row_prefill which takes ks/ke). A caller-provided buffer is used as-is.
    torch::Tensor out_logits =
        out.has_value() ? out.value()
                        : torch::empty({M, N}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
    TORCH_CHECK(out_logits.scalar_type() == torch::kFloat32 && out_logits.is_contiguous() &&
                    out_logits.size(0) == M && out_logits.size(1) == N,
                "out must be a contiguous f32 [M, N] tensor");
    if (M == 0 || N == 0) return out_logits;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(device_of(out_logits));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    dispatch_fp8_mqa_logits(static_cast<const fp8*>(q_fp8.data_ptr()), static_cast<const fp8*>(k_fp8.data_ptr()),
                            kv_scale.data_ptr<float>(), weights.data_ptr<float>(), cu_seqlen_ks.data_ptr<int>(),
                            cu_seqlen_ke.data_ptr<int>(), out_logits.data_ptr<float>(), M, N, split_n, num_warps_,
                            clean_logits, rev > 0, stream);
    return out_logits;
}
