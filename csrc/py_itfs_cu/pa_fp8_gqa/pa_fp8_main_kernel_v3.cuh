// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026 Page_Attetion_GQA_fp8 project
//
// FP8 paged-attention decode main kernel (v3) — cross-iteration K
// software-pipelined variant.
//
// What changed vs v2
// ------------------
// In v2, every kbi iteration starts with:
//   (a) 4 + 4 = 8 global_load_dword for K/V block-table entries
//   (b) 8 buffer_load_dwordx4 for K data
//   (c) 8 buffer_load_dwordx4 for V data
//   (d) QK MFMA chain consuming K data
//   (e) softmax, P->LDS, PV MFMA, running max/sum update
//
// The compiler is forced to insert an `s_waitcnt vmcnt(0)` (full VMEM
// drain) between (a)/(b)/(c) because it can't track per-op completion
// precisely.  Empirically this introduces ~4 vmcnt(0) full drains per
// inner-loop body, costing ~30-80 cycles each.
//
// Gluon avoids this by **prefetching iter (N+1)'s K data at the END of
// iter N's body**, overlapped with the PV MFMA chain.  The first iter's
// K data is prefetched in the kernel prologue.  As a result the loop body
// runs with up to 11 in-flight VMEM operations and never drops to
// vmcnt(0).
//
// v3 mirrors this pattern at the source level:
//   - Persistent state at function scope:
//       int          kphys_num[kTLoop];
//       int          kphys_off[kTLoop];
//       unsigned int v_phys[kNWarps];
//       PaWide       Klocal[kTLoop][kWideQkheLoop];
//   - Prologue: compute_kv_bt(kbi_start) + load_k(Klocal).
//   - Inside loop body: consume Klocal in QK MFMA (no reload here).
//   - After PV MFMA + running update, if (kbi+1 < kbi_stop):
//       compute_kv_bt(kbi+1) + load_k(Klocal)   <-- the cross-iter prefetch
//
// Register pressure: adds ~12 SGPR-class VGPR per lane for kphys_num/off/
// v_phys.  Klocal already existed as a 32-VGPR-per-lane temporary; making
// it function-scope keeps the same allocation but extends its lifetime
// across the iter boundary.  Empirically gfx942 LLVM still settles at
// ≤128 VGPR/wave so launch_bounds(256, 2) (2 waves/SIMD) is preserved.

#pragma once

#include "pa_fp8_common.h"
#include "pa_fp8_main_kernel.cuh"      // v0::... constants
#include "pa_fp8_main_kernel_v2.cuh"   // v2:: constants (kWideQkheLoop, ...)

namespace pa_fp8_gqa {

// ---------------------------------------------------------------------------
// pa_fp8_main_kernel_v3 — cross-iter K prefetch on top of v2 wide-load.
// ---------------------------------------------------------------------------
template <typename output_t, int Mtp>
__global__ __launch_bounds__(v0::kNumThreads, 2)
void pa_fp8_main_kernel_v3(
    const __hip_fp8_e4m3_fnuz* __restrict__ q,
    const __hip_fp8_e4m3_fnuz* __restrict__ k_cache,
    const __hip_fp8_e4m3_fnuz* __restrict__ v_cache,
    const float                              softmax_scale,
    const float* __restrict__                q_scale_ptr,
    const float* __restrict__                k_scale_ptr,
    const float* __restrict__                v_scale_ptr,
    const float* __restrict__                p_scale_ptr,
    const float* __restrict__                p_scale_inv_ptr,
    const bool                               has_p_scale,
    const int* __restrict__                  block_tables,
    const int* __restrict__                  context_lens,
    const int                                max_num_blocks_per_seq,
    const int                                q_stride,
    const int                                kv_block_stride,
    const int                                kv_head_stride,
    float* __restrict__                      exp_sums,
    float* __restrict__                      max_logits,
    output_t* __restrict__                   out,
    const int                                num_kblocks_per_fat_part)
{
    using namespace v0;
    constexpr int kMtp = Mtp;
    static_assert(Mtp == 1 || Mtp == 2, "v3 supports Mtp in {1, 2}");

    constexpr float kLog2E    = 1.4426950408889634f;
    constexpr float kInvLog2E = 0.6931471805599453f;

    const auto seq_idx     = blockIdx.x;
    const auto fp_idx      = blockIdx.y;
    const auto kv_head_idx = blockIdx.z;

    const int warpid   = threadIdx.x / WARP_SIZE;
    const int laneid   = threadIdx.x % WARP_SIZE;
    const int lane16id = laneid % 16;
    const int rowid    = laneid / 16;

    const int num_fat_partitions = gridDim.y;
    const int total_num_heads    = gridDim.z * kGqaRatio;
    const int context_len        = context_lens[seq_idx];
    const int total_num_kblocks  = PAGQA_DIVUP(context_len, kTParSize);

    const int kbi_start    = fp_idx * num_kblocks_per_fat_part;
    const int kbi_stop_raw = kbi_start + num_kblocks_per_fat_part;
    const int kbi_stop     = (kbi_stop_raw < total_num_kblocks)
                                 ? kbi_stop_raw : total_num_kblocks;

    if (kbi_start >= total_num_kblocks) return;

    const int wg_start_head_idx    = kv_head_idx * kGqaRatio;
    const int wg_start_kv_head_idx = kv_head_idx;
    const int num_context_blocks   = PAGQA_DIVUP(context_len, kBlockSize);
    const int last_ctx_block       = num_context_blocks - 1;
    const int* block_table_seq     = block_tables + seq_idx * max_num_blocks_per_seq;

    __shared__ _T8x8 shared_logits[kNWarps * kTLoop * kSlotsPerWarpT];
    __shared__ float shared_qk[kNWarps * 16 * 2];

    const __amdgpu_buffer_rsrc_t k_rsrc =
        pa_make_buffer_rsrc(k_cache + wg_start_kv_head_idx * kv_head_stride);
    const __amdgpu_buffer_rsrc_t v_rsrc =
        pa_make_buffer_rsrc(v_cache + wg_start_kv_head_idx * kv_head_stride);

    const int q_token_for_lane = (kMtp == 1) ? 0 : (lane16id >> 3);
    const int head_for_lane    = lane16id & (kGqaRatio - 1);
    const int q_head_idx       = wg_start_head_idx + head_for_lane;
    const int64_t q_scale_idx =
          (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane)
        * static_cast<int64_t>(total_num_heads) + q_head_idx;
    const float qk_base_log2 = softmax_scale * q_scale_ptr[q_scale_idx] * kLog2E;
    const float v_scale_perhead = v_scale_ptr[kv_head_idx];
    const float p_scale_perhead = has_p_scale ? p_scale_ptr[q_head_idx] : 1.f;
    const float p_scale_inv_perhead =
        has_p_scale ? p_scale_inv_ptr[q_head_idx] : 1.f;

    struct PaWide { int64_t lo; int64_t hi; };
    PaWide Qlocal[v2::kWideQkheLoop];
    {
        const int64_t query_row_off =
            (static_cast<int64_t>(seq_idx) * kMtp + q_token_for_lane) * q_stride
            + (wg_start_head_idx + head_for_lane) * kHeadSize;
        const __hip_fp8_e4m3_fnuz* q_row = q + query_row_off;
        #pragma unroll
        for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
        {
            const int hd_off = qkhe * v2::kK_PER_WIDE_QKHE
                             + rowid * v2::kFp8PerLaneWide;
            const int64_t* p =
                reinterpret_cast<const int64_t*>(q_row + hd_off);
            Qlocal[qkhe].lo = p[0];
            Qlocal[qkhe].hi = p[1];
        }
    }

    float   m_running = -FLT_MAX;
    float   l_running = 0.f;
    floatx4 o_running[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

    // --- Persistent (across-iter) prefetch state. ----------------------
    // Holds K block-table + K data for the kbi iteration that is ABOUT
    // to execute (i.e. the data the next QK MFMA will consume).  Filled
    // by either the prologue (iter kbi_start) or the tail prefetch of
    // the previous iter.
    //
    // Note: `kphysical_block_offset = lane16id` (constant per lane,
    // independent of t and kbi) — proof: kglobal_token_idx
    // = kbi * 256 + warpid * 64 + t * 16 + lane16id, all of which are
    // multiples of 16 except the lane16id term.  We still keep the array
    // form because the buffer_load address computation reads it as
    // `kphys_off[t] * 16`, and the compiler optimises the constant fold.
    int          kphys_num[kTLoop];
    int          kphys_off[kTLoop];
    unsigned int v_phys[kNWarps];
    PaWide       Klocal[kTLoop][v2::kWideQkheLoop];

    constexpr unsigned int kVBytesPerVhe =
        (unsigned int)(kNWarps * 16 * kBlockSize);  // 1024 B
    const unsigned int k_chunk_row_off =
        (unsigned int)rowid * (unsigned int)v2::kBytesPerChunkAllSlot;

    // Compute K BT + V BT for kbi_idx into the persistent state buffers.
    auto compute_kv_bt_for_iter = [&](int kbi_idx) __attribute__((always_inline))
    {
        const int pst = kbi_idx * kTParSize;
        const int pbs = kbi_idx * (kTParSize / kBlockSize);
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            const int klocal_token_idx  = kTokensPerWarp * warpid + t * 16 + lane16id;
            const int kglobal_token_idx = pst + klocal_token_idx;
            const int kblock_idx        = warpid * kTLoop + t;
            const int bt_g_idx          = pbs + kblock_idx;
            const int bt_g_idx_safe     = (bt_g_idx < num_context_blocks)
                                              ? bt_g_idx : last_ctx_block;
            kphys_num[t]   = block_table_seq[bt_g_idx_safe];
            kphys_off[t]   = kglobal_token_idx % kBlockSize;
        }
        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++)
        {
            const int v_bt_g_idx = pbs + v_group * kRowsPerWarp + rowid;
            const int v_bt_g_idx_safe =
                (v_bt_g_idx < num_context_blocks) ? v_bt_g_idx : last_ctx_block;
            v_phys[v_group] = (unsigned int)block_table_seq[v_bt_g_idx_safe];
        }
    };

    // Load K data for all kTLoop tiles into the persistent Klocal buffer
    // using the kphys_num/off currently in registers.
    auto load_k_persist = [&]() __attribute__((always_inline))
    {
        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            const unsigned int kblock_number = (unsigned int)kphys_num[t];
            const unsigned int k_base_voffset =
                  kblock_number * (unsigned int)kv_block_stride
                + (unsigned int)kphys_off[t] * kElems16B_fp8
                + k_chunk_row_off;
            #pragma unroll
            for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
            {
                const unsigned int voff =
                      k_base_voffset
                    + (unsigned int)qkhe * (unsigned int)v2::kBytesPerWideQkhe;
                const pa_u32x4 v = pa_buffer_load_b128(k_rsrc, voff);
                Klocal[t][qkhe].lo = pa_u32x4_low_long(v);
                Klocal[t][qkhe].hi = pa_u32x4_high_long(v);
            }
        }
    };

    // PROLOGUE: prefetch iter kbi_start's K BT + K data so the first
    // pass of the loop body finds Klocal already populated.
    compute_kv_bt_for_iter(kbi_start);
    load_k_persist();

    for (int kbi = kbi_start; kbi < kbi_stop; kbi++)
    {
        const int partition_start_token_idx = kbi * kTParSize;

        if (kbi != kbi_start) __syncthreads();

        floatx4 d_out[kTLoop];
        pa_u32x4 V_wide[kNWarps][kVheLoop];
        {
            auto load_v_slice_wide = [&](int v_group) __attribute__((always_inline))
            {
                const unsigned int v_phys_g = v_phys[v_group];
                const unsigned int v_base_voffset =
                      v_phys_g * (unsigned int)kv_block_stride
                    + (unsigned int)(warpid * 16 + lane16id) * kBlockSize;
                V_wide[v_group][0] = pa_buffer_load_b128_nt(v_rsrc, v_base_voffset);
                V_wide[v_group][1] = pa_buffer_load_b128_nt(
                    v_rsrc, v_base_voffset + kVBytesPerVhe);
            };

            // V loads interleaved with QK MFMA on the prefetched Klocal.
            //
            // Note: there is NO load_k_tile here — Klocal was prefetched
            // either by the prologue or by the previous iter's tail.
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                load_v_slice_wide(t);

                d_out[t] = floatx4{0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int qkhe = 0; qkhe < v2::kWideQkheLoop; qkhe++)
                {
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal[t][qkhe].lo, Qlocal[qkhe].lo, d_out[t]);
                    d_out[t] = pa_mfma16x16x32_fp8_fp8(
                        Klocal[t][qkhe].hi, Qlocal[qkhe].hi, d_out[t]);
                }
                pa_apply_qk_token_scales_for_block(
                    d_out[t], k_scale_ptr, kphys_num[t], rowid * 4,
                    gridDim.z, kv_head_idx, kBlockSize, qk_base_log2);
            }
        }

        const int qkout_token_idx = partition_start_token_idx
                                    + kTokensPerWarp * warpid + rowid * 4;
        float qk_max  = -FLT_MAX;
        float exp_sum = 0.f;
        {
            const int valid_upper = context_len;
            const bool interior_partition =
                (partition_start_token_idx + kTParSize) <= valid_upper;

            if (!interior_partition)
            {
                #pragma unroll
                for (int t = 0; t < kTLoop; t++)
                {
                    const int local_token_idx = qkout_token_idx + t * 16;
                    #pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        if ((local_token_idx + i) >= valid_upper)
                            d_out[t][i] = -FLT_MAX;
                    }
                }
            }

            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    qk_max = fmaxf(qk_max, d_out[t][i]);
            qk_max = fmaxf(qk_max, pa_shfl_xor_32(qk_max));
            qk_max = fmaxf(qk_max, pa_shfl_xor_within_32<16>(qk_max));

            const floatx4 nqk_max{-qk_max, -qk_max, -qk_max, -qk_max};
            floatx4 exp_sum_v4{0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int t = 0; t < kTLoop; t++)
            {
                const floatx4 diff = d_out[t] + nqk_max;
                floatx4 v;
                v[0] = __builtin_amdgcn_exp2f(diff[0]);
                v[1] = __builtin_amdgcn_exp2f(diff[1]);
                v[2] = __builtin_amdgcn_exp2f(diff[2]);
                v[3] = __builtin_amdgcn_exp2f(diff[3]);
                d_out[t]    = v;
                exp_sum_v4 += v;
            }
            exp_sum = exp_sum_v4[0] + exp_sum_v4[1] + exp_sum_v4[2] + exp_sum_v4[3];
            exp_sum = exp_sum + pa_shfl_xor_32(exp_sum);
            exp_sum = exp_sum + pa_shfl_xor_within_32<16>(exp_sum);
        }

        if (laneid < 16)
        {
            const int slot = lane16id * (kNWarps * 2) + warpid * 2;
            shared_qk[slot + 0] = qk_max;
            shared_qk[slot + 1] = exp_sum;
        }
        __syncthreads();

        float partition_qk_max  = -FLT_MAX;
        float partition_exp_sum = 0.f;
        float warp_scale;
        {
            float warp_qk_max[kNWarps];
            float warp_exp_sum[kNWarps];
            const int base = lane16id * (kNWarps * 2);
            #pragma unroll
            for (int w = 0; w < kNWarps; w++)
            {
                warp_qk_max[w]   = shared_qk[base + w * 2 + 0];
                warp_exp_sum[w]  = shared_qk[base + w * 2 + 1];
                partition_qk_max = fmaxf(partition_qk_max, warp_qk_max[w]);
            }
            float warp_qk_max_exp[kNWarps];
            #pragma unroll
            for (int w = 0; w < kNWarps; w++)
            {
                warp_qk_max_exp[w] = __builtin_amdgcn_exp2f(
                    warp_qk_max[w] - partition_qk_max);
                partition_exp_sum += warp_exp_sum[w] * warp_qk_max_exp[w];
            }
            warp_scale = warp_qk_max_exp[warpid];
        }

        #pragma unroll
        for (int t = 0; t < kTLoop; t++)
        {
            d_out[t] *= warp_scale * p_scale_perhead;
            const uint32_t pk = pa_pk_fp8x4(
                d_out[t][0], d_out[t][1], d_out[t][2], d_out[t][3]);
            const int idx = v0::shared_logits_index(warpid, t, lane16id, rowid);
            shared_logits[idx].i64 = static_cast<int64_t>(pk);
        }
        __syncthreads();

        floatx4 pv_acc[kVheLoop];
        #pragma unroll
        for (int vhe = 0; vhe < kVheLoop; vhe++)
            pv_acc[vhe] = floatx4{0.f, 0.f, 0.f, 0.f};

        int64_t P_lo_per_g[kNWarps];
        int64_t P_hi_per_g[kNWarps];
        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++)
        {
            _T8x8 P_lo_pack, P_hi_pack;
            P_lo_pack.b8x4[0] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 0)].i64);
            P_lo_pack.b8x4[1] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 1)].i64);
            P_hi_pack.b8x4[0] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 2)].i64);
            P_hi_pack.b8x4[1] = static_cast<uint32_t>(
                shared_logits[v0::shared_logits_index(v_group, rowid, lane16id, 3)].i64);
            P_lo_per_g[v_group] = P_lo_pack.i64;
            P_hi_per_g[v_group] = P_hi_pack.i64;
        }

        // --- TAIL PREFETCH for next iter --------------------------------
        // Compute kbi+1's K BT + V BT and issue the 8 K dwordx4 loads.
        // Placing this BEFORE the PV MFMA chain (rather than after) gives
        // the longest possible memory-latency hiding window: the 8 K
        // dwordx4 loads can stream while the 16 PV MFMA instructions
        // execute.  By the time iter (kbi+1)'s QK MFMA chain begins,
        // most (ideally all) K loads have already landed.
        //
        // Klocal IS dead at this point — last consumer was the QK MFMA
        // chain in lines above (already retired into d_out[t]).
        const bool has_next_iter = (kbi + 1 < kbi_stop);
        if (has_next_iter)
        {
            compute_kv_bt_for_iter(kbi + 1);
            load_k_persist();
        }

        #pragma unroll
        for (int v_group = 0; v_group < kNWarps; v_group++)
        {
            const int64_t P_lo = P_lo_per_g[v_group];
            const int64_t P_hi = P_hi_per_g[v_group];
            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
            {
                const pa_u32x4 V_chunk = V_wide[v_group][vhe];
                const int64_t V_lo = pa_u32x4_low_long(V_chunk);
                const int64_t V_hi = pa_u32x4_high_long(V_chunk);
                pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(V_lo, P_lo, pv_acc[vhe]);
                pv_acc[vhe] = pa_mfma16x16x32_fp8_fp8(V_hi, P_hi, pv_acc[vhe]);
            }
        }

        {
            const float m_new = fmaxf(m_running, partition_qk_max);
            const float alpha = (m_running > -FLT_MAX)
                ? __builtin_amdgcn_exp2f(m_running - m_new) : 0.f;
            const float beta  = __builtin_amdgcn_exp2f(partition_qk_max - m_new);
            l_running = alpha * l_running + beta * partition_exp_sum;
            #pragma unroll
            for (int vhe = 0; vhe < kVheLoop; vhe++)
                o_running[vhe] = alpha * o_running[vhe] + beta * pv_acc[vhe];
            m_running = m_new;
        }
    }

    const float inv_l = __fdividef(1.f, l_running + 1e-6f);
    const float post_scale = inv_l * v_scale_perhead * p_scale_inv_perhead;
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        o_running[vhe] *= post_scale;

    if (warpid == 0 && rowid == 0 && lane16id < kMtp * kGqaRatio)
    {
        const int head_idx = lane16id & (kGqaRatio - 1);
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        const int64_t maxp = static_cast<int64_t>(num_fat_partitions);
        const int64_t offset =
              static_cast<int64_t>(query_start_off + q_token_for_lane)
                  * static_cast<int64_t>(total_num_heads) * maxp
            + (static_cast<int64_t>(wg_start_head_idx) + head_idx) * maxp
            + static_cast<int64_t>(fp_idx);
        max_logits[offset] = m_running * kInvLog2E;
        exp_sums[offset]   = l_running;
    }

    _B16x4 outelems[kVheLoop];
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
        outelems[vhe] = pa_from_floatx4<output_t>(o_running[vhe]);

    __syncthreads();
    #pragma unroll
    for (int vhe = 0; vhe < kVheLoop; vhe++)
    {
        const int idx = v0::shared_logits_index(warpid, vhe, lane16id, rowid);
        _T8x8 cell;
        cell.b16x4 = outelems[vhe];
        shared_logits[idx] = cell;
    }
    __syncthreads();

    if (warpid == 0)
    {
        const int64_t query_start_off = static_cast<int64_t>(seq_idx) * kMtp;
        constexpr int kGqa4_  = (kGqaRatio + 3) / 4;
        constexpr int kRowsHere = kMtp * kGqa4_;

        const int head_elem_idx = lane16id * 8;
        if (head_elem_idx < kHeadSize)
        {
            const int64_t hsz_maxp_mult =
                static_cast<int64_t>(kHeadSize)
              * static_cast<int64_t>(num_fat_partitions);
            #pragma unroll
            for (int local_row_idx = 0; local_row_idx < kRowsHere; local_row_idx++)
            {
                const int q_tok_w  = (kMtp == 1) ? 0 : (local_row_idx >> 1);
                const int head_quad = local_row_idx & 1;
                const int local_head_idx_in_quad = rowid;
                const int packed_lane = q_tok_w * 8 + head_quad * 4 + local_head_idx_in_quad;
                _B16x8 vout;
                const int offset1 = (head_elem_idx / 16) % 4;
                const int offset2 = head_elem_idx / 16 / kNWarps;
                const int offset3 = (head_elem_idx / 4) % 4;
                #pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    const int idx =
                        v0::shared_logits_index(offset1, offset2, packed_lane, offset3 + i);
                    vout.xy[i] = shared_logits[idx].b16x4;
                }
                const int head_idx = head_quad * 4 + local_head_idx_in_quad;
                if (head_idx < kGqaRatio)
                {
                    const int64_t out_head_idx =
                        static_cast<int64_t>(wg_start_head_idx + head_idx);
                    output_t* out_ptr = out
                        + (query_start_off + q_tok_w) * total_num_heads * hsz_maxp_mult
                        + out_head_idx * hsz_maxp_mult
                        + fp_idx * kHeadSize
                        + head_elem_idx;
                    *reinterpret_cast<_B16x8*>(out_ptr) = vout;
                }
            }
        }
    }
}

} // namespace pa_fp8_gqa
