// MFMA tile GEMM0/1 main path (asm cl_gemm0/cl_gemm1 + pi ping-pong).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/kernels/pa_decode_device_utils.hpp"
#include "opus_pa/kernels/pa_gemm0_utils.hpp"
#include "opus_pa/kernels/pa_gemm1_utils.hpp"
#include "opus_pa/kernels/pa_mfma_layout_utils.hpp"
#include "opus_pa/kernels/pa_softmax_utils.hpp"
#include "opus_pa/kernels/pa_v_layout_utils.hpp"

namespace pa_decode {

static constexpr int kPiCount = 2;
static constexpr int kKvRegTotalDwords = kSzVk * kPiCount;  // _v_K_reg_size = 64

template<int GQA, int SUB_KV, int HEAD_DIM, int KV_REG_DWORDS, int BLOCK_SIZE>
__device__ __forceinline__ void load_kv_scale_pi(const float* kq_base,
                                                   const float* vq_base,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   int kv_nheads,
                                                   int kv_head_idx,
                                                   int pi,
                                                   float k_scale_pi[SUB_KV / 2],
                                                   float v_scale_pi[SUB_KV / 2]) {
    constexpr int kHalf = SUB_KV / 2;
    constexpr int kBlksPerPi = SUB_KV / BLOCK_SIZE / 2;
    for (int i = threadIdx.x; i < kHalf; i += blockDim.x) {
        const int blk_local = i / BLOCK_SIZE;
        const int in_blk = i % BLOCK_SIZE;
        const int bt_slot = blk_local + pi * kBlksPerPi;
        float ks = 1.f;
        float vs = 1.f;
        if (kq_base != nullptr && bt_slot < valid_blks) {
            const uint32_t page = page_ids[bt_slot];
            ks = kv_scale_at(kq_base, page, in_blk, BLOCK_SIZE, kv_nheads, kv_head_idx);
            vs = kv_scale_at(vq_base, page, in_blk, BLOCK_SIZE, kv_nheads, kv_head_idx);
        }
        k_scale_pi[i] = ks;
        v_scale_pi[i] = vs;
    }
    __syncthreads();
}

__device__ __forceinline__ uint64_t pack_u64_bytes(const uint8_t b[8]) {
    const uint32_t lo = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
                        (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    const uint32_t hi = static_cast<uint32_t>(b[4]) | (static_cast<uint32_t>(b[5]) << 8) |
                        (static_cast<uint32_t>(b[6]) << 16) | (static_cast<uint32_t>(b[7]) << 24);
    return pack_u64(lo, hi);
}

// GEMM0 S = Q @ K^T via fp8 MFMA, built from gathered operands (probe-verified layout):
//   A = Q  -> A[m][k]: m = query (lane%16),        k = head = kk*32 + 8*(lane/16) + i
//   B = K  -> B[k][n]: n = kv-within-16 (lane%16), k = head
// D[m=query][n=kv] matches mfma_scatter_scores_slice (row=query, col=kv). Raw fp8 dot;
// q_deq/k_scale applied afterwards by scores_apply_qk_dequant.
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void build_scores_mfma_pi(const uint8_t* q_fp8_tile,
                                                     const uint8_t* k_pool,
                                                     const uint32_t* page_ids,
                                                     int valid_blks,
                                                     uint32_t stride_blk,
                                                     uint32_t stride_kvhead,
                                                     int kv_head_idx,
                                                     int pi,
                                                     int kv_offset,
                                                     int tile_kv,
                                                     float s_out[GQA][SUB_KV]) {
    constexpr int kNumJ = SUB_KV / 64;   // 4 MFMA N-blocks per pi
    constexpr int kNumK = HEAD_DIM / 32; // 4 K-steps (head chunks of 32)

    const int lane = lane_id();
    const int wave = wave_id();
    const int query = lane & 15;
    const int g = lane >> 4;
    // page_ids and s_out are per-tile; use LOCAL kv indices (kv_offset is the global
    // tile base, only needed by the tail mask for ctx_len comparison).
    (void)kv_offset;
    const int kv_pi_base = pi * (SUB_KV / 2);
    const int bound = tile_kv;

#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int kv_base = kv_pi_base + j * 64;
        const int kv_lane = kv_base + (wave << 4) + query;  // this lane's kv (n = lane%16)
        mfma_acc4 acc{};
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
        for (int kk = 0; kk < kNumK; ++kk) {
            uint8_t qb[8];
            uint8_t kb[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int head = kk * 32 + g * 8 + i;
                qb[i] = (query < GQA) ? q_fp8_tile[query * HEAD_DIM + head] : 0;
                kb[i] = (kv_lane < bound)
                            ? k_hbm_tile_byte<HEAD_DIM, BLOCK_SIZE>(k_pool, page_ids, valid_blks,
                                                                    stride_blk, stride_kvhead,
                                                                    kv_head_idx, kv_lane, head)
                            : static_cast<uint8_t>(0);
            }
            acc = mfma_fp8_fp8_step(acc, pack_u64_bytes(qb), pack_u64_bytes(kb), kk == 0);
        }
#endif
        mfma_scatter_scores_slice<GQA, SUB_KV>(acc, lane, wave, 0, kv_base, bound, s_out);
    }
    __syncthreads();
}

template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int NUM_PAIRS, int LOAD_INSTS,
         int IMM_STRIDE>
__device__ __forceinline__ void load_k_regs_tile(const uint8_t* k_pool,
                                                 const uint32_t* page_ids,
                                                 int valid_blks,
                                                 uint32_t stride_blk,
                                                 uint32_t stride_kvhead,
                                                 int lane,
                                                 int wave,
                                                 int tg_idx,
                                                 int bt_slots,
                                                 uint32_t k_regs[KV_REG_DWORDS * kPiCount]) {
    (void)wave;
    U64 k_pool_u = u64_from_ptr(k_pool);
    U64 k_combined =
        u64_add_imm(k_pool_u, k_pool_lane_base_offset(lane, tg_idx, stride_kvhead));
    U64 k_addrs[NUM_PAIRS];
#pragma unroll
    for (int fch = 0; fch < kPiCount; ++fch) {
        const uint32_t* pi_pages = v_mem_load_page_ids_slice<SUB_KV, BLOCK_SIZE>(page_ids, fch);
        const int pi_valid = v_mem_load_valid_blks_slice<SUB_KV, BLOCK_SIZE>(valid_blks, fch);
        k_mem_va_upd<NUM_PAIRS>(k_addrs, k_combined, pi_pages, lane, wave, bt_slots, pi_valid,
                                stride_blk);
        k_mem_load<LOAD_INSTS, KV_REG_DWORDS, IMM_STRIDE>(k_regs, k_addrs, fch, KV_REG_DWORDS);
    }
}

template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int NUM_PAIRS, int LOAD_INSTS,
         int IMM_STRIDE>
__device__ __forceinline__ void load_v_regs_tile(const uint8_t* v_pool,
                                                 const uint32_t* page_ids,
                                                 int valid_blks,
                                                 uint32_t stride_blk,
                                                 uint32_t stride_kvhead,
                                                 int lane,
                                                 int wave,
                                                 int tg_idx,
                                                 int bt_slots,
                                                 uint32_t v_regs[KV_REG_DWORDS * kPiCount]) {
    (void)SUB_KV;
    (void)HEAD_DIM;
    U64 v_pool_u = u64_from_ptr(v_pool);
    U64 v_combined =
        u64_add_imm(v_pool_u, v_pool_lane_base_offset(lane, wave, tg_idx, stride_kvhead));
    U64 v_addrs[NUM_PAIRS];
#pragma unroll
    for (int fch = 0; fch < kPiCount; ++fch) {
        const uint32_t* pi_pages = v_mem_load_page_ids_slice<SUB_KV, BLOCK_SIZE>(page_ids, fch);
        const int pi_valid = v_mem_load_valid_blks_slice<SUB_KV, BLOCK_SIZE>(valid_blks, fch);
        v_mem_va_upd<NUM_PAIRS>(v_addrs, v_combined, pi_pages, lane, wave, bt_slots, pi_valid,
                                stride_blk);
        v_mem_load<LOAD_INSTS, KV_REG_DWORDS, IMM_STRIDE>(v_regs, v_addrs, fch, KV_REG_DWORDS);
    }
}

// Accumulate MFMA GEMM1 head slice into O (multiply by per-query p_deq after MFMA).
// asm cl_gemm1(pi): vV_off = pi*sz_vV + j*sz_vV/2 + k*vs_AB, k in [0, 128/32).
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS>
__device__ __forceinline__ void gemm1_mfma_pi(const uint8_t (*p_fp8)[SUB_KV],
                                              const uint32_t* v_regs,
                                              const uint32_t p_regs[16],
                                              int pi,
                                              int query_base,
                                              const float* p_deq_scales,
                                              float o_out[GQA][HEAD_DIM],
                                              const uint8_t* v_pool,
                                              const uint32_t* page_ids,
                                              int valid_blks,
                                              uint32_t stride_blk,
                                              uint32_t stride_kvhead,
                                              int kv_head_idx,
                                              int block_size,
                                              int tile_kv,
                                              int lane,
                                              int wave) {
    constexpr int kHalf = SUB_KV / 2;
    constexpr int kNumJ = HEAD_DIM / 64;
    constexpr int kNumK = kHalf / 32;  // asm: 128/32 per pi
    constexpr int kPiKvSlice = 32;

    const int kv_pi_base = pi * kHalf;

#pragma unroll
    for (int j = 0; j < kNumJ; ++j) {
        const int head_base = j * 64;
        const int vV_base = pi * KV_REG_DWORDS + j * kSzVvHalf;
        mfma_acc4 acc{};
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
        for (int kk = 0; kk < kNumK; ++kk) {
            const int kv_base = kv_pi_base + kk * kPiKvSlice;
            uint32_t p_lo = 0;
            uint32_t p_hi = 0;
            uint32_t v_lo = 0;
            uint32_t v_hi = 0;
#if PA_GEMM1_LEGACY_PRESHAPE
            p_lo = p_regs[kk * kVsAb + 0];
            p_hi = p_regs[kk * kVsAb + 1];
#else
            p_mfma_gather_a_pair_wave<GQA, SUB_KV, kPiKvSlice>(p_fp8, kv_base, lane, wave, p_lo,
                                                               p_hi);
#endif
#if defined(PA_GEMM1_VGATHER)
            v_mfma_gather_b_pair<SUB_KV, HEAD_DIM, BLOCK_SIZE>(
                v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, kv_head_idx, block_size,
                tile_kv, kv_base, head_base, lane, wave, v_lo, v_hi);
#else
            const int vV_off = vV_base + kk * kVsAb;
            v_lo = v_regs[vV_off + 0];
            v_hi = v_regs[vV_off + 1];
#endif
            const uint64_t v_pk = pack_u64(v_lo, v_hi);
            const uint64_t p_pk = pack_u64(p_lo, p_hi);
            acc = mfma_fp8_fp8_gemm1_step(acc, p_pk, v_pk, kk == 0);
        }
#endif
        mfma_acc4 scaled{};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = 0;
            int col = 0;
            mfma16_lane_to_mn(lane, k, row, col);
            (void)col;
            const int qi = query_base + row;
            const float deq = (qi < GQA) ? p_deq_scales[qi] : 0.f;
            scaled[k] = acc[k] * deq;
        }
        mfma_gather_o_slice_accum<GQA, HEAD_DIM>(scaled, lane, wave, query_base, head_base, o_out);
    }

    __syncthreads();
}

template<int GQA, int SUB_KV, int HEAD_DIM, int KV_REG_DWORDS, int NUM_PAIRS, int LOAD_INSTS,
         int IMM_STRIDE>
__device__ __forceinline__ void gemm1_prepare_p_merged(
    const uint8_t p_fp8[GQA][SUB_KV],
    int pi,
    uint32_t* dyn_resp_lds,
    int lane,
    int wave,
    uint32_t p_merged[16]) {
    constexpr int kHalf = SUB_KV / 2;
    constexpr int kPiK = kHalf / 32;
    uint32_t p_pi[16];
    p_prepare_mfma_regs_slice<GQA, SUB_KV>(p_fp8, pi * kHalf, dyn_resp_lds, lane, wave, p_pi);
#pragma unroll
    for (int kk = 0; kk < kPiK; ++kk) {
        const int gk = pi * kPiK + kk;
        p_merged[gk * kVsAb + 0] = p_pi[kk * kVsAb + 0];
        p_merged[gk * kVsAb + 1] = p_pi[kk * kVsAb + 1];
    }
}

// Bisect path: one-shot asm cl_gemm1 (8 K-steps) with merged P + full V register tile.
template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int NUM_PAIRS,
         int LOAD_INSTS, int IMM_STRIDE>
__device__ __forceinline__ void gemm1_clgemm1_tile(const uint8_t p_fp8[GQA][SUB_KV],
                                                    const uint8_t* v_pool,
                                                    const uint32_t* page_ids,
                                                    int valid_blks,
                                                    uint32_t stride_blk,
                                                    uint32_t stride_kvhead,
                                                    int bt_slots,
                                                    int tg_idx,
                                                    const float* p_deq_scales,
                                                    uint32_t* dyn_resp_lds,
                                                    int lane,
                                                    int wave,
                                                    float o_out[GQA][HEAD_DIM]) {
    uint32_t v_regs[KV_REG_DWORDS * kPiCount];
    load_v_regs_tile<SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                     IMM_STRIDE>(
        v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, lane, wave, tg_idx, bt_slots,
        v_regs);

    uint32_t p_merged[16] = {};
    gemm1_prepare_p_merged<GQA, SUB_KV, HEAD_DIM, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                           IMM_STRIDE>(p_fp8, 0, dyn_resp_lds, lane, wave, p_merged);
    __syncthreads();
    gemm1_prepare_p_merged<GQA, SUB_KV, HEAD_DIM, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                           IMM_STRIDE>(p_fp8, 1, dyn_resp_lds, lane, wave, p_merged);
    __syncthreads();

    mfma_acc4 r_mfma[HEAD_DIM / 64];
    cl_gemm1_fp8<SUB_KV, HEAD_DIM, KV_REG_DWORDS>(v_regs, p_merged, r_mfma);
    gemm1_clgemm1_gather_o<GQA, HEAD_DIM>(r_mfma, 0, p_deq_scales, o_out);
}

#if defined(PA_GEMM1_BISECT)
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void gemm1_bisect_report(const float o_mfma[GQA][HEAD_DIM],
                                                    const float o_ref[GQA][HEAD_DIM],
                                                    float* dbg) {
    __shared__ float bisect_smem[256];
    gemm1_max_abs_diff<GQA, HEAD_DIM>(o_mfma, o_ref, bisect_smem);
    if (threadIdx.x == 0 && dbg != nullptr) {
        dbg[0] = bisect_smem[0];
    }
    __syncthreads();
}
#endif

#if defined(PA_V_REGS_BISECT)
template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int NUM_PAIRS, int LOAD_INSTS,
         int IMM_STRIDE>
__device__ __forceinline__ void v_regs_bisect_tile(const uint8_t* v_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int block_size,
                                                   int tile_kv,
                                                   int lane,
                                                   int wave,
                                                   int tg_idx,
                                                   int bt_slots,
                                                   float* dbg) {
    uint32_t v_regs[KV_REG_DWORDS * 2];
    load_v_regs_tile<SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                     IMM_STRIDE>(
        v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, lane, wave, tg_idx, bt_slots,
        v_regs);
    __shared__ float bisect_smem[256 * 8];
    v_regs_lane_bisect<SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS, LOAD_INSTS, NUM_PAIRS,
                       IMM_STRIDE>(
        v_regs, v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, block_size, tile_kv,
        lane, wave, tg_idx, bt_slots, bisect_smem);
    __syncthreads();
    v_regs_bisect_reduce(bisect_smem, dbg);
}
#endif

template<int GQA, int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int NUM_PAIRS,
         int LOAD_INSTS, int IMM_STRIDE>
__device__ __forceinline__ void gemm1_mfma_tile(const uint8_t p_fp8[GQA][SUB_KV],
                                                 const float (*p_f32)[SUB_KV],
                                                 const uint8_t* v_pool,
                                                 const uint32_t* page_ids,
                                                 int valid_blks,
                                                 uint32_t stride_blk,
                                                 uint32_t stride_kvhead,
                                                 int block_size,
                                                 int tile_kv,
                                                 int bt_slots,
                                                 int tg_idx,
                                                 int kv_head_idx,
                                                 const float* p_deq_scales,
                                                 uint32_t* dyn_resp_lds,
                                                 int lane,
                                                 int wave,
                                                 float o_out[GQA][HEAD_DIM],
                                                 float* dbg) {
#if defined(PA_GEMM1_BISECT)
    float o_ref[GQA][HEAD_DIM];
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        o_ref[idx / HEAD_DIM][idx % HEAD_DIM] = 0.f;
    }
    __syncthreads();
#else
    (void)dbg;
#endif
    (void)p_f32;

#if defined(PA_GEMM1_USE_CLGEMM)
    gemm1_clgemm1_tile<GQA, SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                      IMM_STRIDE>(
        p_fp8, v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, bt_slots, tg_idx,
        p_deq_scales, dyn_resp_lds, lane, wave, o_out);
#if defined(PA_GEMM1_BISECT)
    gemm1_pv_reference<GQA, SUB_KV, HEAD_DIM>(p_fp8, v_pool, page_ids, valid_blks, stride_blk,
                                              stride_kvhead, kv_head_idx, block_size, tile_kv,
                                              p_deq_scales, nullptr, o_ref);
    gemm1_bisect_report<GQA, HEAD_DIM>(o_out, o_ref, dbg);
#endif
    return;
#endif

    constexpr int kHalf = SUB_KV / 2;
    uint32_t v_regs[KV_REG_DWORDS * kPiCount] = {};
#if !defined(PA_GEMM1_VGATHER)
    // With VGATHER the V operand is gathered from HBM; the asm V register global-load
    // port is disabled (its OOB reads could fault).
    load_v_regs_tile<SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS, NUM_PAIRS, LOAD_INSTS,
                     IMM_STRIDE>(
        v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, lane, wave, tg_idx, bt_slots,
        v_regs);
#endif

#if defined(PA_GEMM1_BISECT)
    for (int idx = threadIdx.x; idx < GQA * HEAD_DIM; idx += blockDim.x) {
        o_ref[idx / HEAD_DIM][idx % HEAD_DIM] = 0.f;
    }
    __syncthreads();
    gemm1_pv_reference<GQA, SUB_KV, HEAD_DIM>(p_fp8, v_pool, page_ids, valid_blks, stride_blk,
                                              stride_kvhead, kv_head_idx, block_size, tile_kv,
                                              p_deq_scales, nullptr, o_ref);
#endif

    for (int pi = 0; pi < kPiCount; ++pi) {
        const int kv_offset = pi * kHalf;
        uint32_t p_regs[16] = {};
#if PA_GEMM1_LEGACY_PRESHAPE
        p_prepare_mfma_regs_slice<GQA, SUB_KV>(p_fp8, kv_offset, dyn_resp_lds, lane, wave,
                                               p_regs);
        __syncthreads();
#endif
#if defined(PA_GEMM1_VREF)
        gemm1_pv_reference_pi_slice<GQA, SUB_KV, HEAD_DIM>(
            p_fp8, v_pool, page_ids, valid_blks, stride_blk, block_size, kv_offset, kHalf, tile_kv,
            stride_kvhead, kv_head_idx, p_deq_scales, o_out);
#else
        gemm1_mfma_pi<GQA, SUB_KV, HEAD_DIM, BLOCK_SIZE, KV_REG_DWORDS>(
            p_fp8, v_regs, p_regs, pi, 0, p_deq_scales, o_out, v_pool, page_ids, valid_blks,
            stride_blk, stride_kvhead, kv_head_idx, block_size, tile_kv, lane, wave);
#endif
    }

#if defined(PA_GEMM1_BISECT)
    gemm1_bisect_report<GQA, HEAD_DIM>(o_out, o_ref, dbg);
#else
    (void)dbg;
#endif
}

}  // namespace pa_decode
