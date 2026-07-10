// asm-structural PA decode kernel (BF16 Q, FP8 KV, GQA8, 2TG x 4W).
// Control flow mirrors PA_A16W8_Q8_2TG_4W_16mx1_64nx4:
//   prologue -> core_loop(cl_p) per wave group -> tail -> R_div_L -> write_out
#pragma once

#include <cmath>
#include <cstring>

#include "opus_pa/pa_decode_defs.h"
#include "opus_pa/kernels/pa_decode_device_utils.hpp"
#include "opus_pa/kernels/pa_q_swizzle_utils.hpp"
#include "opus_pa/kernels/pa_gemm0_utils.hpp"
#include "opus_pa/kernels/pa_softmax_utils.hpp"
#include "opus_pa/kernels/pa_mfma_tile_utils.hpp"
#include "opus_pa/kernels/pa_scores_compact_utils.hpp"
#include "opus_pa/kernels/pa_q_gemm_utils.hpp"
#include "opus_pa/kernels/pa_output_utils.hpp"
#if defined(PA_DEBUG_SCORES)
#include "opus_pa/kernels/pa_scores_debug_utils.hpp"
#endif

namespace pa_decode {
namespace detail {

template<typename Traits>
__device__ __forceinline__ void core_loop_tile(int cl_p,
                               int tile,
                               int kv_offset,
                               int tile_kv,
                               uint32_t ctx_len,
                               int valid_blks,
                               const pa_decode_kargs& kargs,
                               const uint32_t* page_ids,
                               int bt_slots,
                               uint32_t stride_blk,
                               float scale_log2e,
                               int kv_head_idx,
                               int lane,
                               int wave,
                               const uint32_t q_mfma_regs[8],
                               float* q_deq_scales,
                               float* fa_max,
                               float* L_acc,
                               float* delta_scale,
                               float (*o_acc)[Traits::HEAD_DIM],
                               float* k_scale_pi0,
                               float* k_scale_pi1,
                               float* v_scale_pi0,
                               float* v_scale_pi1,
                               float (*s_sparse)[Traits::SUB_KV],
                               float (*s_dense)[Traits::SUB_KV],
                               const uint8_t* q_fp8_tile,
                               const bf16_t* q_lds,
                               int q_row_stride_elems,
                               uint8_t* k_tile,
                               uint8_t (*p_fp8)[Traits::SUB_KV],
                               float* p_deq_scales,
                               float (*o_tile)[Traits::HEAD_DIM],
                               uint32_t* dyn_resp_lds,
                               uint32_t k_regs[Traits::KV_REG_DWORDS * 2]) {
    (void)cl_p;
    (void)tile;

    const int kv_nheads = static_cast<int>(kargs.kv_nheads);
    const float* kq_base = static_cast<const float*>(kargs.ptr_KQ);
    const float* vq_base = static_cast<const float*>(kargs.ptr_VQ);

    if (tile > 0) {
        pa_r_procss_rescale<Traits::GQA_RATIO, Traits::HEAD_DIM>(o_acc, delta_scale);
    }

    load_kv_scale_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM,
                     Traits::KV_REG_DWORDS, Traits::BLOCK_SIZE>(
        kq_base, vq_base, page_ids, valid_blks, kv_nheads, kv_head_idx, 0, k_scale_pi0,
        v_scale_pi0);
    load_kv_scale_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM,
                     Traits::KV_REG_DWORDS, Traits::BLOCK_SIZE>(
        kq_base, vq_base, page_ids, valid_blks, kv_nheads, kv_head_idx, 1, k_scale_pi1,
        v_scale_pi1);

    // GEMM0 MFMA builds A/B via gather (build_scores_mfma_pi); the asm K register
    // global-load port is disabled (its OOB reads could fault alongside the V-reg load).
    (void)k_regs;

#if defined(PA_V_REGS_BISECT)
    v_regs_bisect_tile<Traits::SUB_KV, Traits::HEAD_DIM, Traits::BLOCK_SIZE,
                       Traits::KV_REG_DWORDS, 4, Traits::KV_LOAD_INSTS, Traits::KV_IMM_STRIDE>(
        static_cast<const uint8_t*>(kargs.ptr_V), page_ids, valid_blks, stride_blk,
        kargs.stride_kvhead, Traits::BLOCK_SIZE, tile_kv, lane, wave, kv_head_idx, bt_slots,
        static_cast<float*>(kargs.ptr_DBG));
    return;
#endif

    // GEMM0 (Q@Kᵀ) via asm MFMA gather. D[query][kv] is already the dense score
    // layout, so scatter straight into s_dense.
    (void)s_sparse;
    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::SUB_KV; idx += blockDim.x) {
        s_dense[idx / Traits::SUB_KV][idx % Traits::SUB_KV] = 0.f;
    }
    __syncthreads();

    build_scores_mfma_pi<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM,
                         Traits::BLOCK_SIZE>(
        q_fp8_tile, static_cast<const uint8_t*>(kargs.ptr_K), page_ids, valid_blks, stride_blk,
        kargs.stride_kvhead, kv_head_idx, 0, kv_offset, tile_kv, s_dense);

    if (tile_kv < Traits::SUB_KV) {
        pa_tail_mask_dense<Traits::GQA_RATIO, Traits::SUB_KV>(s_dense, kv_offset, tile_kv,
                                                              ctx_len);
    }

    scores_apply_qk_dequant<Traits::GQA_RATIO, Traits::SUB_KV>(
        s_dense, tile_kv, q_deq_scales, k_scale_pi0, k_scale_pi1);

    if (tile_kv < Traits::SUB_KV) {
        pa_tail_mask_dense<Traits::GQA_RATIO, Traits::SUB_KV>(s_dense, kv_offset, tile_kv,
                                                              ctx_len);
    }

#if defined(PA_DUMP_SCORES)
    if (kargs.ptr_DBG != nullptr) {
        float* dbg = static_cast<float*>(kargs.ptr_DBG);
        for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * tile_kv; idx += blockDim.x) {
            const int g = idx / tile_kv;
            const int k = idx % tile_kv;
            dbg[g * Traits::SUB_KV + k] = s_dense[g][k];
        }
        __syncthreads();
    }
#endif

    pa_fuse_alu_tile_scales<Traits::GQA_RATIO, Traits::SUB_KV>(
        s_dense, tile_kv, q_deq_scales, k_scale_pi0, k_scale_pi1, v_scale_pi0, v_scale_pi1,
        scale_log2e, fa_max, L_acc, delta_scale, p_fp8, p_deq_scales);

    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::HEAD_DIM; idx += blockDim.x) {
        o_tile[idx / Traits::HEAD_DIM][idx % Traits::HEAD_DIM] = 0.f;
    }
    __syncthreads();

    // GEMM1 (P@V) via asm MFMA with V register gather.
    gemm1_mfma_tile<Traits::GQA_RATIO, Traits::SUB_KV, Traits::HEAD_DIM, Traits::BLOCK_SIZE,
                    Traits::KV_REG_DWORDS, 4, Traits::KV_LOAD_INSTS, Traits::KV_IMM_STRIDE>(
        p_fp8, nullptr, static_cast<const uint8_t*>(kargs.ptr_V), page_ids, valid_blks, stride_blk,
        kargs.stride_kvhead, Traits::BLOCK_SIZE, tile_kv, bt_slots, kv_head_idx, kv_head_idx,
        p_deq_scales, dyn_resp_lds, lane, wave, o_tile, nullptr);

    o_acc_add_tile<Traits::GQA_RATIO, Traits::HEAD_DIM>(o_acc, o_tile);
}

}  // namespace detail
}  // namespace pa_decode

template<typename Traits>
__device__ void pa_decode_kernel_body(const pa_decode_kargs& kargs) {
    using namespace pa_decode;

    constexpr int kBtSlots = Traits::SUB_KV / Traits::BLOCK_SIZE;
    (void)kBtSlots;

    const int batch_id = static_cast<int>(blockIdx.y);
    const int kv_head_idx = static_cast<int>(blockIdx.x);
    const int lane = lane_id();
    const int wave = wave_id();
    const int cl_p = (wave < 2) ? 0 : 1;
    (void)cl_p;

    const uint32_t stride_blk = kargs.stride_blk;
    const float scale_log2e = kargs.scale_log2e;

    const uint32_t* context_lens = static_cast<const uint32_t*>(kargs.ptr_CL);
    const uint32_t kv_seq = context_lens[batch_id];
    const int aligned_kv = static_cast<int>(kv_seq & ~static_cast<uint32_t>(Traits::SUB_KV - 1));
    const int num_aligned_tiles = aligned_kv / Traits::SUB_KV;
    const int tail_kv = static_cast<int>(kv_seq - static_cast<uint32_t>(aligned_kv));

    const uint32_t q_batch_stride = kargs.stride_Q_batch;
    const uint8_t* q_byte_base = static_cast<const uint8_t*>(kargs.ptr_Q);
    q_byte_base += kv_head_idx * kargs.stride_Q + batch_id * q_batch_stride;

    bf16_t* out_base = static_cast<bf16_t*>(kargs.ptr_O);
    out_base += kv_head_idx * kargs.stride_Q / static_cast<uint32_t>(sizeof(bf16_t)) +
                batch_id * q_batch_stride / static_cast<uint32_t>(sizeof(bf16_t));

    constexpr int kQRowStrideElems = Traits::Q_LDS_ROW_ELEMS;
    __shared__ bf16_t q_lds[Traits::Q_LDS_ROWS * kQRowStrideElems];
    uint32_t* dyn_resp_lds_ptr = nullptr;
#if defined(PA_MFMA_GEMM0) || defined(PA_MFMA_GEMM1)
    __shared__ uint32_t dyn_resp_lds[kDynRspDwords];
    for (int i = threadIdx.x; i < kDynRspDwords; i += blockDim.x) {
        dyn_resp_lds[i] = 0;
    }
    __syncthreads();
    dyn_resp_lds_ptr = dyn_resp_lds;
#endif

    load_q_tile_to_shared<Traits::SUB_Q, Traits::HEAD_DIM, Traits::GQA_RATIO,
                          Traits::Q_LDS_ROWS, Traits::Q_LDS_ROW_ELEMS>(
        reinterpret_cast<const bf16_t*>(q_byte_base), q_lds, kQRowStrideElems);

    __shared__ float q_deq_scales[Traits::GQA_RATIO];
    __shared__ uint8_t q_fp8_tile[Traits::GQA_RATIO * Traits::HEAD_DIM];

    q_rowmajor_fp8_per_query_from_lds<Traits::GQA_RATIO, Traits::HEAD_DIM>(
        q_lds, kQRowStrideElems, q_fp8_tile, q_deq_scales);

    __shared__ float fa_max[Traits::GQA_RATIO];
    __shared__ float L_acc[Traits::GQA_RATIO];
    __shared__ float delta_scale[Traits::GQA_RATIO];
    __shared__ uint8_t p_fp8[Traits::GQA_RATIO][Traits::SUB_KV];
    __shared__ float p_deq_scales[Traits::GQA_RATIO];
    __shared__ float o_tile[Traits::GQA_RATIO][Traits::HEAD_DIM];
    __shared__ float o_acc[Traits::GQA_RATIO][Traits::HEAD_DIM];
    __shared__ float k_scale_pi0[Traits::SUB_KV / 2];
    __shared__ float k_scale_pi1[Traits::SUB_KV / 2];
    __shared__ float v_scale_pi0[Traits::SUB_KV / 2];
    __shared__ float v_scale_pi1[Traits::SUB_KV / 2];
    __shared__ uint32_t page_ids[kBtSlots];
    __shared__ uint32_t pages_snap[kBtSlots];
    __shared__ float s_sparse[Traits::GQA_RATIO][Traits::SUB_KV];
    __shared__ float s_dense[Traits::GQA_RATIO][Traits::SUB_KV];
    __shared__ uint8_t k_tile_buf[Traits::GEMM0_KV_SLICE * Traits::HEAD_DIM];
    uint8_t* k_tile = k_tile_buf;

    for (int g = threadIdx.x; g < Traits::GQA_RATIO; g += blockDim.x) {
        fa_max[g] = kNegInf;
        L_acc[g] = 0.f;
        delta_scale[g] = 1.f;
    }
    for (int idx = threadIdx.x; idx < Traits::GQA_RATIO * Traits::HEAD_DIM; idx += blockDim.x) {
        o_acc[idx / Traits::HEAD_DIM][idx % Traits::HEAD_DIM] = 0.f;
    }
    __syncthreads();

    uint32_t k_regs[Traits::KV_REG_DWORDS * 2];
    uint32_t q_mfma_regs[8] = {};

#if defined(PA_MFMA_GEMM0)
    q_swizzle_pipeline_paref_deq<Traits::GQA_RATIO, Traits::HEAD_DIM>(
        q_lds, lane, wave, q_deq_scales, q_mfma_regs, dyn_resp_lds_ptr);
    __syncthreads();
#endif

    for (int tile = 0; tile < num_aligned_tiles; ++tile) {
        const int kv_offset = tile * Traits::SUB_KV;
        int valid_blks = 0;
        load_block_table_tile_offset<Traits::SUB_KV, Traits::BLOCK_SIZE>(
            static_cast<const uint32_t*>(kargs.ptr_BT), batch_id, kargs.max_blks, kv_seq, kv_offset,
            page_ids, valid_blks);
        for (int i = threadIdx.x; i < kBtSlots; i += blockDim.x) {
            pages_snap[i] = page_ids[i];
        }
        __syncthreads();

        detail::core_loop_tile<Traits>(cl_p, tile, kv_offset, Traits::SUB_KV, kv_seq, valid_blks,
                                    kargs, pages_snap, kBtSlots, stride_blk, scale_log2e, kv_head_idx,
                                    lane, wave, q_mfma_regs, &q_deq_scales[0], &fa_max[0], &L_acc[0],
                                    &delta_scale[0], &o_acc[0], &k_scale_pi0[0], &k_scale_pi1[0], &v_scale_pi0[0],
                                    &v_scale_pi1[0], &s_sparse[0], &s_dense[0], q_fp8_tile, q_lds,
                                    Traits::Q_LDS_ROW_ELEMS, k_tile, &p_fp8[0], &p_deq_scales[0],
                                    &o_tile[0], dyn_resp_lds_ptr, k_regs);
    }

    if (tail_kv > 0) {
        const int kv_offset = aligned_kv;
        int valid_blks = 0;
        load_block_table_tile_offset<Traits::SUB_KV, Traits::BLOCK_SIZE>(
            static_cast<const uint32_t*>(kargs.ptr_BT), batch_id, kargs.max_blks, kv_seq, kv_offset,
            page_ids, valid_blks);
        for (int i = threadIdx.x; i < kBtSlots; i += blockDim.x) {
            pages_snap[i] = page_ids[i];
        }
        __syncthreads();

        detail::core_loop_tile<Traits>(cl_p, num_aligned_tiles, kv_offset, tail_kv, kv_seq, valid_blks,
                                    kargs, pages_snap, kBtSlots, stride_blk, scale_log2e, kv_head_idx,
                                    lane, wave, q_mfma_regs, &q_deq_scales[0], &fa_max[0], &L_acc[0],
                                    &delta_scale[0],
                                    &o_acc[0], &k_scale_pi0[0], &k_scale_pi1[0], &v_scale_pi0[0],
                                    &v_scale_pi1[0], &s_sparse[0], &s_dense[0], q_fp8_tile, q_lds,
                                    Traits::Q_LDS_ROW_ELEMS, k_tile, &p_fp8[0], &p_deq_scales[0],
                                    &o_tile[0], dyn_resp_lds_ptr, k_regs);
    }

    r_write_out_bf16<Traits::GQA_RATIO, Traits::HEAD_DIM>(&o_acc[0], &L_acc[0], out_base,
                                                          Traits::HEAD_DIM);
}
