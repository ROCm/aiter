// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused FP8 MQA logits + top-k (gfx950): the scan kernel, a copy of mqa_v35wes::kernel
// (csrc/kernels/fp8_mqa_logits.cu) with two output modes:
//
//   MODE_DENSE  fp32 logits stored exactly like v35wes (no -inf fill), row stride `out_cols`. Used for the sample
//               pass (the keys j*S gathered into their own buffer).
//   MODE_EMIT   candidate emission: a logit is a candidate when res >= tau[row]; candidate i of row m (split s) is
//               the u64 (n << 32) | bits(res) at cand[cand_index(s, m, i)] (interleaved pair segments, 2*CAP slots
//               per row pair), positions from ballot/mbcnt and wave-uniform per-row counters. Exactly ONE buffer
//               store per pair (non-candidate lanes -> out-of-range offset; i >= CAP = overflow, dropped by the range
//               check), so the hand-placed vmcnt accounting of v35wes (2*SUB stores per stage) is unchanged.
//               cnt[split*M + m] = candidates seen (> CAP = overflow).
//
// Geometry, pipeline, epilogue and schedule are v35wes's: 8 waves x 4 rows = 32 rows per WG, 2-stage LDS ring of
// 2 x 32 tokens, PF = 4 register prefetch slots, hand-placed vmcnt, scalar epilogue co-issued under the MFMAs,
// deferred row-pair tails, early stage barrier + s_setprio staircase, clean/masked stage split shared over SplitN.
// Build flags: -fno-honor-nans -fno-slp-vectorize -mllvm -disable-machine-sink (see fp8_mqa_logits.cu).
#pragma once
#include "fmt_common.cuh"

namespace mqa_topk {

constexpr int MODE_DENSE = 0;
constexpr int MODE_EMIT = 1;

#define MQA_TOPK_SYNC_TILE() __syncthreads()

template <int NUM_WARPS, int SUB, int PF, int MODE, bool REVERSE_ROWS>
__global__ __launch_bounds__(NUM_WARPS * WARP)
void scan_kernel(const fp8* __restrict__ Q_ptr, const fp8* __restrict__ K_ptr, const float* __restrict__ kv_scale_ptr,
                 const float* __restrict__ W_ptr, const int* __restrict__ ks_ptr, const int* __restrict__ ke_ptr,
                 int M, int N, int SplitN,
                 // MODE_DENSE
                 float* __restrict__ out_ptr, int out_cols,
                 // MODE_EMIT
                 const float* __restrict__ tau_ptr, unsigned long long* __restrict__ cand_ptr, int* __restrict__ cnt_ptr,
                 int CAP)
{
    constexpr int BLOCK_M    = 4;
    constexpr int ROWS_WG    = BLOCK_M * NUM_WARPS;
    constexpr int TILE_N     = 32;
    constexpr int TILE_BYTES = TILE_N * HEAD_SIZE;          // 4096 per sub-tile
    constexpr int STAGE_TOK  = TILE_N * SUB;
    constexpr int STAGE_BYTES= TILE_BYTES * SUB;
    constexpr int CHUNKS     = STAGE_BYTES / 16;            // 16-B chunks per stage
    constexpr int NTHREADS   = NUM_WARPS * WARP;
    constexpr int STAGES     = 2;
    constexpr int NACC       = 16;
    constexpr bool EMIT      = (MODE == MODE_EMIT);
    using VecOutMFMA = __attribute__((__vector_size__(NACC * sizeof(float)))) float;

    __shared__ __attribute__((aligned(16))) fp8   k_lds[STAGES][STAGE_BYTES];
    __shared__ __attribute__((aligned(16))) float sc_lds[STAGES][STAGE_TOK];
    constexpr int SC_WAVES = STAGE_TOK / WARP;               // waves that load the stage's scales (1 lane = 1 token)
    static_assert(STAGE_TOK % WARP == 0 && SC_WAVES <= NUM_WARPS, "scale producer mapping");

    const int tid  = threadIdx.x;
    const int l    = tid % WARP;
    const int warp = __builtin_amdgcn_readfirstlane(tid / WARP);
    const int bx   = REVERSE_ROWS ? (gridDim.x - 1 - blockIdx.x) : blockIdx.x;
    const int m0wg = bx * ROWS_WG;
    const int m0   = m0wg + warp * BLOCK_M;
    const int pid_split = blockIdx.y;
    if (m0wg >= M) return;

    const int NT = N;   // token space of the pipeline

    const int rw   = l / 16;
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
            w2[r][2 * j]     = f32x2{wv[0], wv[1]};
            w2[r][2 * j + 1] = f32x2{wv[2], wv[3]};
        }
    }
    // ---- WG-wide tile range (uniform across waves) ----
    int lo = INT_MAX, hi = 0, max_ks = 0, min_ke = INT_MAX;
    for (int r = 0; r < ROWS_WG; ++r) {
        const int m = m0wg + r;
        if (m < M) { const int a = ks_ptr[m], b = ke_ptr[m]; lo = min(lo, a); hi = max(hi, b); max_ks = max(max_ks, a); min_ke = min(min_ke, b); }
    }
    lo = __builtin_amdgcn_readfirstlane(max(lo, 0)); hi = __builtin_amdgcn_readfirstlane(min(hi, NT));
    max_ks = __builtin_amdgcn_readfirstlane(max_ks); min_ke = __builtin_amdgcn_readfirstlane(min(min_ke, NT));
    const bool empty = (lo >= hi);
    lo = empty ? 0 : (lo & ~(TILE_N - 1));
    const int num_tiles   = empty ? 0 : MQA_TOPK_CDIV(hi - lo, STAGE_TOK);   // stages
    // clean stage range [tc0, tc1): every token of the stage is inside [ks,ke) of ALL rows of the WG, all rows exist, < NT
    const bool all_rows = (m0wg + ROWS_WG) <= M;
    int tc0 = min(MQA_TOPK_CDIV(max(max_ks - lo, 0), STAGE_TOK), num_tiles);
    int tc1 = all_rows ? min(max((min_ke - lo) / STAGE_TOK, 0), num_tiles) : 0;
    tc1 = max(tc1, tc0);
    const int n_masked = tc0 + (num_tiles - tc1);               // stages [0,tc0) U [tc1,num_tiles), contiguous index space
    const float NEG_INF = -__builtin_huge_valf();
    (void)NEG_INF;

    if (empty) {
        if constexpr (EMIT) {
            if (l == 0)
                for (int r = 0; r < BLOCK_M; ++r) if (m0 + r < M) cnt_ptr[(int64_t)pid_split * M + m0 + r] = 0;
        }
        return;
    }

    // ---- buffer resources ----
    const i32x4 k_v = raw_rsrc(K_ptr, N * HEAD_SIZE);
    const i32x4 s_v = raw_rsrc(kv_scale_ptr, N * 4);
    // output: DENSE -> fp32 rows (stride out_cols); EMIT -> the interleaved segment of each row pair (rows r, r+1 =
    // m0 + r even): 2 * CAP u64 slots, row r at even slots, row r + 1 at odd slots (see cand_index()); a slot index
    // >= 2 * CAP (row count >= CAP = overflow) is dropped by the buffer range check.
    const int Mp = M + (M & 1);
    __amdgpu_buffer_rsrc_t o_rsrc[BLOCK_M / 2];
    __amdgpu_buffer_rsrc_t c_rsrc[BLOCK_M / 2];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; r += 2) {
        if constexpr (EMIT) {
            const int row = __builtin_amdgcn_readfirstlane(m0 + r);   // < Mp, even
            c_rsrc[r / 2] = __builtin_amdgcn_make_buffer_rsrc((void*)(cand_ptr + ((int64_t)pid_split * Mp + row) * CAP), 0, 2 * CAP * 8, MQA_TOPK_BUF_RSRC_WORD3);
            o_rsrc[r / 2] = __builtin_amdgcn_make_buffer_rsrc((void*)cand_ptr, 0, 0, MQA_TOPK_BUF_RSRC_WORD3);  // dummy stores only
        } else {
            const int row = __builtin_amdgcn_readfirstlane(min(m0 + r, M - 1));
            o_rsrc[r / 2] = __builtin_amdgcn_make_buffer_rsrc((void*)(out_ptr + (int64_t)row * out_cols), 0, 2 * out_cols * 4, MQA_TOPK_BUF_RSRC_WORD3);
        }
    }
    const int o_voff = hset * out_cols * 4 + jtok * 4;
    (void)o_voff;
    const int hmask = -hset;                     // EMIT: per-half position correction
    float tau_reg[BLOCK_M];
    int cnt_row[BLOCK_M] = {0, 0, 0, 0};   // wave-uniform
    if constexpr (EMIT) {
        #pragma unroll
        for (int r = 0; r < BLOCK_M; ++r) tau_reg[r] = (m0 + r < M) ? tau_ptr[m0 + r] : 0.f;
    }

    // ---- producer mapping: chunk c (16 B) of the tile: token c/8, 16-B piece c%8 ----
    // Each lane copies CPL chunks; the LDS destination XOR-swizzles the piece index by ((token >> 1) & 7).
    constexpr int CPL = MQA_TOPK_CDIV(CHUNKS, NTHREADS);
    static_assert(CHUNKS % NTHREADS == 0, "every lane issues exactly CPL K loads per stage (vmcnt accounting)");
    // vmcnt accounting (per wave, lower bound: K loads + output stores; wave 0's scale loads only add younger ops)
    constexpr int ST_PER_STAGE = 2 * SUB;                       // pair stores per stage
    constexpr int OPS_STAGE    = CPL + ST_PER_STAGE;
    constexpr int W_PRO        = (PF - 1) * CPL;                 // prologue: slot 0, younger = slots 1..PF-1
    constexpr int W_STEADY     = ST_PER_STAGE + (PF - 1) * OPS_STAGE;
    static_assert(W_STEADY < 64, "vmcnt field is 6 bits");
    int   pf_lds_off[CPL];
    int   pf_voff[CPL];
    #pragma unroll
    for (int i = 0; i < CPL; ++i) {
        const int c = tid + i * NTHREADS;
        const int t = c / 8, pc = c % 8;            // t = token within stage (0..STAGE_TOK-1)
        pf_voff[i]    = c * 16;
        pf_lds_off[i] = t * HEAD_SIZE + ((pc ^ ((t >> 1) & 7)) * 16);
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
    auto mfma_a = [&](int r, const Bop& B, VecOutMFMA& c) {
        VecOutMFMA z = {};
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][0], B.b0, z, 0, 0, 0, 0, 0, 0);
    };
    auto mfma_b = [&](int r, const Bop& B, VecOutMFMA& c) {
        c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(q_reg[r][1], B.b1, c, 0, 0, 0, 0, 0, 0);
    };
    // half H of row r: elements 8H..8H+7; even/odd chains (fixed rounding order, bit-exact with v35wes)
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
    float sel_tau[BLOCK_M / 2];
    #pragma unroll
    for (int r = 0; r < BLOCK_M; r += 2) {
        sel_ks[r / 2] = hset ? ks_reg[r + 1] : ks_reg[r];
        sel_ke[r / 2] = hset ? ke_reg[r + 1] : ke_reg[r];
        sel_ok[r / 2] = (m0 + r + hset < M);
        if constexpr (EMIT) sel_tau[r / 2] = hset ? tau_reg[r + 1] : tau_reg[r];
    }
    const unsigned OOB_VOFF = 0x7ffffff0u;   // beyond num_records -> store dropped by the range check
    // pair_finish: everything after the permlane (p0 / p1 already swapped); pair_store = swap + finish
    auto pair_finish = [&](int r, float p0, float p1, float scale, int token_base, auto masked_c) {
        constexpr bool MASKED = decltype(masked_c)::value;
        const float res = (p0 + p1) * scale;
        if constexpr (EMIT) {
            // Branch-free (every ?: operand is a named value; bool & instead of &&: clang turns ?: on expressions into
            // exec branches that split the sched_barrier-pinned sub-tile and spill). Review variant "EV4":
            // per-row SGPR counters, ONE full-wave mbcnt with a per-half correction, interleaved pair slots.
            const int abs_pos = token_base + jtok;
            bool cand = res >= sel_tau[r / 2];
            if constexpr (MASKED) cand = cand & (sel_ok[r / 2] != 0) & (abs_pos < NT) & (abs_pos >= sel_ks[r / 2]) & (abs_pos < sel_ke[r / 2]);
            const uint64_t msk = __ballot(cand);
            const unsigned mlo = (unsigned)msk, mhi = (unsigned)(msk >> 32);
            const unsigned mb = __builtin_amdgcn_mbcnt_hi(mhi, __builtin_amdgcn_mbcnt_lo(mlo, 0u));
            const int c0 = cnt_row[r], c1 = cnt_row[r + 1] - __builtin_popcount(mlo);   // SALU
            const int base = c0 + ((c1 - c0) & hmask);
            const unsigned vt = ((mb + (unsigned)base) * 2u + (unsigned)hset) << 3;
            const unsigned voff = cand ? vt : 0x40800000u;                                // 4.0f: inline constant, out of range
            cnt_row[r] += __builtin_popcount(mlo);
            cnt_row[r + 1] += __builtin_popcount(mhi);
            using u32x2 = __attribute__((__vector_size__(8))) unsigned int;
            __builtin_amdgcn_raw_buffer_store_b64(u32x2{__builtin_bit_cast(unsigned, res), (unsigned)abs_pos}, c_rsrc[r / 2], voff, 0, 0);
        } else if constexpr (!MASKED) {
            __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, res), o_rsrc[r / 2], (unsigned)o_voff, token_base * 4, 0);
        } else {
            const int abs_pos = token_base + jtok;
            const bool valid  = abs_pos >= sel_ks[r / 2] && abs_pos < sel_ke[r / 2];
            const bool write  = sel_ok[r / 2] && abs_pos < NT;
            const unsigned voff = (write && valid) ? (unsigned)o_voff : OOB_VOFF;
            __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, res), o_rsrc[r / 2], voff, token_base * 4, 0);
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
            const int sc_soff = token_base * 4;
            const int sc_vo = tid * 4;                           // computed here: clang does not capture plain
            const i32x4 sv = s_v;                                // variables named only in asm operands
            asm volatile("buffer_load_dword %0, %1, %2, %3 offen" : "=v"(ps) : "v"(sc_vo), "s"(sv), "s"(sc_soff) : "memory");
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
    // drain: wait for the clamped, never-used inline-asm prefetches before their VGPRs can be reused. The compiler does
    // not track inline-asm load destinations, so it may otherwise reuse them while the loads are still in flight.
    static_assert(CPL == 1, "drain operand list assumes one chunk per lane");
    auto drain_prefetch = [&]() {
        if constexpr (PF == 4)
            asm volatile("s_waitcnt vmcnt(0)" :: "v"(pf_reg0[0]), "v"(pf_reg1[0]), "v"(pf_reg2[0]), "v"(pf_reg3[0]),
                         "v"(pf_sc0), "v"(pf_sc1), "v"(pf_sc2), "v"(pf_sc3) : "memory");
        else
            asm volatile("s_waitcnt vmcnt(0)" :: "v"(pf_reg0[0]), "v"(pf_reg1[0]), "v"(pf_sc0), "v"(pf_sc1) : "memory");
    };
#define FLUSH_PEND() { drain_prefetch(); \
                       pair_finish(0, pd_P0, pd_P1, Bo[1].scale, pd_tok, MK{}); \
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
    // (RD_NEXT = the read_lds call for the next sub-tile; issued between mfma_a(3) and the tail rows).
    // Wait-state bookkeeping: see fp8_mqa_logits.cu (every permlane input written >= 2 instructions earlier).
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
    // one LDS stage: SUB sub-tiles; early barrier + priority staircase as in v35wes.
#define NEXT_STAGE_READ(NXT) read_lds(Bo[0], IC<NXT>{}, IC<0>{})
#define STAGE_STEP(CUR, NXT, WS, TT, MK, WC, P0)                                                          \
    {                                                                                                     \
        const int stage_base = __builtin_amdgcn_readfirstlane(tokbase(TT));                              \
        write_lds(IC<WS>{}, IC<NXT>{}, IC<WC>{});                                                        \
        issue_loads(min((TT) + PF + 1, te - 1), IC<WS>{});                                               \
        if constexpr (SUB == 2) {                                                                         \
            SUBTILE(0, 1, stage_base, read_lds(Bo[1], IC<CUR>{}, IC<1>{}), MK, P0, 1, 0, MQA_TOPK_SYNC_TILE()) \
            SUBTILE(1, 0, stage_base + TILE_N, NEXT_STAGE_READ(NXT), MK, true, 3, 2, (void)0)            \
        } else {                                                                                          \
            SUBTILE(0, 1, stage_base, read_lds(Bo[1], IC<CUR>{}, IC<1>{}), MK, P0, 1, -1, (void)0)       \
            SUBTILE(1, 0, stage_base + TILE_N, read_lds(Bo[0], IC<CUR>{}, IC<2>{}), MK, true, 0, -1, (void)0) \
            SUBTILE(0, 1, stage_base + 2 * TILE_N, read_lds(Bo[1], IC<CUR>{}, IC<3>{}), MK, true, 3, -1, MQA_TOPK_SYNC_TILE()) \
            SUBTILE(1, 0, stage_base + 3 * TILE_N, NEXT_STAGE_READ(NXT), MK, true, 2, -1, (void)0)        \
        }                                                                                                 \
    }

    static_assert(SUB == 2 || SUB == 4, "SUB = 2 or 4");
    static_assert(PF == 2 || PF == 4, "PF = 2 or 4 register slots");
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
    auto share = [&](int n, int& b_, int& e_) { const int per = MQA_TOPK_CDIV(n, SplitN); b_ = pid_split * per; e_ = min(n, b_ + per); };
    int mb, me, cb, ce;
    share(n_masked, mb, me);
    share(tc1 - tc0, cb, ce); cb += tc0; ce += tc0;
    run_pipeline(std::true_type{}, mb, me);
    wait_vmcnt<0>();   // no asm prefetch load may still be in flight when the next pipeline reuses its VGPRs
    run_pipeline(std::false_type{}, cb, ce);
    wait_vmcnt<0>();
    if constexpr (EMIT) {
        if (l == 0) {
            #pragma unroll
            for (int r = 0; r < BLOCK_M; ++r) if (m0 + r < M) cnt_ptr[(int64_t)pid_split * M + m0 + r] = cnt_row[r];
        }
    }
}

}  // namespace mqa_topk
