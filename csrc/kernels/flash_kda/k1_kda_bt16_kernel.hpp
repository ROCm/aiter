// FlashKDA K1 (prepare) — HIP/MFMA, CHUNK=16, D=128, one wavefront per block.
// Grid (NT, B*H); block 64. Produces the six workspace intermediates consumed
// by K2: k_decayed, q_decayed, k_restored, g_total, INV=(I-L)^-1, Mqk.
// Math mirrors tests/torch_ref.py exactly (per chunk, per head).
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

// Build the varlen tile prefix-sum: tile_prefix[i] = sum_{j<i} ceil(seq_len_j/16).
// tile_prefix[N] is the exact total chunk-tile count (<= total_tiles upper bound).
// The context affine/hybrid route repurposes segment_prefix as its filtered
// GROUP_CHUNKS prefix; direct and all non-context routes retain the ordinary
// C64 segment prefix.  Single block, one thread walks the small cu_seqlens once.
__global__ void k1_build_tile_prefix(
        const int32_t* __restrict__ cu_seqlens, int N,
        int* __restrict__ tile_prefix, int* __restrict__ pair_prefix,
        int* __restrict__ segment_prefix,
        int context_group_chunks,
        int context_direct_max_chunks,
        const void* __restrict__ init_state,
        void* __restrict__ final_state,
        int H,
        bool initialize_empty_state,
        bool has_state_in,
        bool has_state_out,
        bool state_fp32) {
    if (threadIdx.x == 0) {
        int acc = 0, pair_acc = 0, seg_acc = 0, context_acc = 0;
        tile_prefix[0] = 0;
        pair_prefix[0] = 0;
        segment_prefix[0] = 0;
        for (int i = 0; i < N; i++) {
            const int64_t slen = int64_t(cu_seqlens[i + 1]) -
                                 int64_t(cu_seqlens[i]);
            const int chunks = int((slen + 16 - 1) / 16);
            acc += chunks;
            pair_acc += (chunks + 1) / 2;
            seg_acc += (chunks + 3) / 4;
            if (context_group_chunks > 0 &&
                (context_direct_max_chunks == 0 ||
                 chunks > context_direct_max_chunks)) {
                context_acc +=
                    (chunks + context_group_chunks - 1) /
                    context_group_chunks;
            }
            tile_prefix[i + 1] = acc;
            pair_prefix[i + 1] = pair_acc;
            segment_prefix[i + 1] = context_group_chunks > 0
                ? context_acc : seg_acc;
        }
    }

    // C-split grids retain one sequence-owned scan CTA even when that
    // sequence has zero segments, but the scan returns before its state
    // load/store.  Fuse the recurrent identity into this already-required
    // prefix launch instead of adding a memset/copy launch or inflating every
    // hot C-split specialization.  Non-C-split routes keep this branch off.
    if (!initialize_empty_state || !has_state_out)
        return;
    constexpr int D = 128;
    const int tid = int(threadIdx.x);
    const int threads = int(blockDim.x);
    for (int seq = 0; seq < N; ++seq) {
        if (cu_seqlens[seq] != cu_seqlens[seq + 1])
            continue;
        const int64_t elements = int64_t(H) * D * D;
        const int64_t base = int64_t(seq) * elements;
        if (state_fp32) {
            const auto* source =
                reinterpret_cast<const uint32_t*>(init_state);
            auto* destination = reinterpret_cast<uint32_t*>(final_state);
            for (int64_t local = tid; local < elements; local += threads)
                destination[base + local] =
                    has_state_in ? source[base + local] : uint32_t{0};
        } else {
            const auto* source =
                reinterpret_cast<const uint16_t*>(init_state);
            auto* destination = reinterpret_cast<uint16_t*>(final_state);
            for (int64_t local = tid; local < elements; local += threads)
                destination[base + local] =
                    has_state_in ? source[base + local] : uint16_t{0};
        }
    }
}

// All pointers are the pre-sliced workspace bases (see launcher).
// Non-varlen (VL=false): grid (NT, B*H); ht = bh*NT + nt.
// Varlen    (VL=true):  grid (total_tiles, H); global tile blockIdx.x is mapped
//   to (seq_idx, local_t) via a binary search on tile_prefix; ht = h*total_tiles
//   + global_tile_idx. Gap tiles (>= tile_prefix[N]) early-return.
template <bool VL = false, bool GVA = false>
__global__ void __launch_bounds__(64)
k1_kda_bt16_kernel(
        const __bf16* __restrict__ q_g,     // [T_total, H_q, D]
        const __bf16* __restrict__ k_g,
        const __bf16* __restrict__ g_g,     // [T_total, H, D]
        const float*  __restrict__ beta_g,  // [T_total, H]
        const float*  __restrict__ A_log_g, // [H]
        const float*  __restrict__ dt_bias_g, // [H, D]
        float scale, float gate_scale,
        int T_seq, int H, int H_q,
        __bf16* __restrict__ ws_kd,   // [n_ht, 16, 128]
        __bf16* __restrict__ ws_qd,
        __bf16* __restrict__ ws_kr,
        float*  __restrict__ ws_gt,   // [n_ht, 128]
        __bf16* __restrict__ ws_inv,  // [n_ht, 16, 16]
        __bf16* __restrict__ ws_mqk,
        const int32_t* __restrict__ cu_seqlens,  // varlen only
        const int* __restrict__ tile_prefix,     // varlen only [N+1]
        int N_seq,                                // varlen only (# sequences)
        int total_tiles) {                        // varlen only (ht column pitch)
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;      // 0..63
    int h, ht, t0, alen;
    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        // binary search on tile_prefix[0..N_seq]: largest seq_idx with
        // tile_prefix[seq_idx] <= gti. Gap tiles (gti >= tile_prefix[N_seq]) fall
        // to seq_idx = N_seq-1 and are dropped by the local_t >= t_tiles guard.
        int lo = 0, hi = N_seq;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
        }
        const int seq_idx = lo;
        const int tiles_before = tile_prefix[seq_idx];
        const int local_t = gti - tiles_before;
        const int64_t bos = cu_seqlens[seq_idx];
        const int64_t eos = cu_seqlens[seq_idx + 1];
        const int seq_len = int(eos - bos);
        const int t_tiles = (seq_len + C - 1) / C;
        if (local_t >= t_tiles) return;   // gap tile past this sequence's chunks
        ht  = h * total_tiles + gti;
        t0  = int(bos) + local_t * C;
        alen = min(C, seq_len - local_t * C);
    } else {
        const int nt = blockIdx.x, bh = blockIdx.y;
        const int NT = gridDim.x;
        const int b = bh / H; h = bh % H;
        ht = bh * NT + nt;
        t0 = b * T_seq + nt * C;
        alen = min(C, T_seq - nt * C);   // rows with real data
    }
    const int qh = GVA ? h / (H / H_q) : h;

    __shared__ __bf16   knorm[C * D];
    __shared__ __bf16   qd[C * D];      // q_norm -> q_decayed
    __shared__ float    gc[C * D];      // g_act -> g_cumsum
    __shared__ __bf16   kd[C * D];
    __shared__ __bf16   kinv[C * D];
    __shared__ __bf16   kr[C * D];
    __shared__ float    g_total[D];
    __shared__ float    beta_act[C];
    __shared__ _Float16 Lm[C * C];
    __shared__ _Float16 INV[C * C];
    __shared__ _Float16 Lk[C * C];      // scratch: L2 / L4 / L8
    __shared__ __bf16   Mqk[C * C];

    // ---- load raw q,k into LDS (bf16), zero pad; load+activate g into gc ----
    const float a_head = ex2(A_log_g[h] * KDA_LOG2E);
    for (int idx = lane; idx < C * D; idx += 64) {
        int m = idx / D, d = idx % D;
        if (m < alen) {
            const int64_t qk_offset =
                (int64_t(t0 + m) * H_q + qh) * D + d;
            const int64_t vg_offset =
                (int64_t(t0 + m) * H + h) * D + d;
            knorm[idx] = k_g[qk_offset];
            qd[idx]    = q_g[qk_offset];
            float gf = bf16_to_f32(g_g[vg_offset]) + dt_bias_g[h * D + d];
            gc[idx]  = gate_scale * sigmoid_tanh(a_head * gf);
        } else {
            knorm[idx] = (__bf16)0.0f; qd[idx] = (__bf16)0.0f; gc[idx] = 0.0f;
        }
    }
    if (lane < C) beta_act[lane] = (lane < alen)
        ? sigmoid_tanh(beta_g[(t0 + lane) * H + h]) : 0.0f;
    __syncthreads();

    // ---- L2 normalize q,k per row (D=128), fp32 accumulate ----
    for (int m = 0; m < C; m++) {
        int d0 = lane, d1 = lane + 64;   // 2 elems/lane cover 128
        float kv0 = bf16_to_f32(knorm[m*D+d0]), kv1 = bf16_to_f32(knorm[m*D+d1]);
        float qv0 = bf16_to_f32(qd[m*D+d0]),    qv1 = bf16_to_f32(qd[m*D+d1]);
        float ks = wave_reduce_sum(kv0*kv0 + kv1*kv1);
        float qs = wave_reduce_sum(qv0*qv0 + qv1*qv1);
        float ki = rsqrtf(ks + 1e-6f), qi = rsqrtf(qs + 1e-6f);
        knorm[m*D+d0] = f32_to_bf16(kv0*ki); knorm[m*D+d1] = f32_to_bf16(kv1*ki);
        qd[m*D+d0]    = f32_to_bf16(qv0*qi); qd[m*D+d1]    = f32_to_bf16(qv1*qi);
    }
    __syncthreads();

    // ---- inclusive cumsum over rows, per column; g_total = last row ----
    // 64 lanes stride over all 128 columns (d = lane, lane+64); each column
    // is visited exactly once.
    for (int d = lane; d < D; d += 64) {
        float acc = 0.0f;
        for (int m = 0; m < C; m++) { acc += gc[m*D + d]; gc[m*D + d] = acc; }
        g_total[d] = acc;
    }
    __syncthreads();

    // ---- decay operands (bf16, matching reference rounding order) ----
    const __bf16 scale_bf = f32_to_bf16(scale);
    for (int idx = lane; idx < C * D; idx += 64) {
        int d = idx % D;
        float gcmd = gc[idx];
        __bf16 dec_p = f32_to_bf16(ex2(gcmd));
        __bf16 dec_n = f32_to_bf16(ex2(-gcmd));
        __bf16 dec_t = f32_to_bf16(ex2(g_total[d]));
        float kn = bf16_to_f32(knorm[idx]);
        float qn = bf16_to_f32(qd[idx]);
        __bf16 kd_v   = f32_to_bf16(kn * bf16_to_f32(dec_p));
        __bf16 kinv_v = f32_to_bf16(kn * bf16_to_f32(dec_n));
        kd[idx]   = kd_v;
        kinv[idx] = kinv_v;
        kr[idx]   = f32_to_bf16(bf16_to_f32(kinv_v) * bf16_to_f32(dec_t));
        __bf16 qtmp = f32_to_bf16(qn * bf16_to_f32(dec_p));
        qd[idx]   = f32_to_bf16(bf16_to_f32(qtmp) * bf16_to_f32(scale_bf));
    }
    __syncthreads();

    // ---- L = tril(kd @ kinv^T, -1) * beta  (fp16);  Mqk = tril(qd @ kinv^T) (bf16) ----
    f32x4 cL = gemm_contract_last<__bf16, D>(kd, kinv, lane);
    f32x4 cM = gemm_contract_last<__bf16, D>(qd, kinv, lane);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int m = (lane >> 4) * 4 + i, n = lane & 15;
        _Float16 lval = (m > n) ? f32_to_f16(cL[i]) * f32_to_f16(beta_act[m]) : (_Float16)0.0f;
        Lm[m*C + n] = lval;
        Mqk[m*C + n] = (m >= n) ? f32_to_bf16(cM[i]) : (__bf16)0.0f;
    }
    __syncthreads();

    // ---- Neumann inverse: INV=I-L; INV+=INV@L^2; INV+=INV@L^4; INV+=INV@L^8 ----
    for (int idx = lane; idx < C * C; idx += 64) {
        int m = idx / C, n = idx % C;
        INV[idx] = (_Float16)((m == n ? 1.0f : 0.0f)) - Lm[idx];
    }
    __syncthreads();
    // Lk = L^2
    { f32x4 c = gemm_std_f16(Lm, Lm, lane); store_acc_16x16(Lk, c, lane); }
    __syncthreads();
    // INV += INV @ L^2
    { f32x4 c = gemm_std_f16(INV, Lk, lane); __syncthreads();
      for (int i=0;i<4;i++){int m=(lane>>4)*4+i,n=lane&15; INV[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    // Lk = L^4 = L^2 @ L^2  (recompute L^2 into a temp via Lm? we overwrote nothing of Lm)
    // Need L^2 still: recompute from Lm each level to avoid extra buffers.
    { f32x4 c2 = gemm_std_f16(Lm, Lm, lane); store_acc_16x16(Lk, c2, lane); }  // Lk=L^2
    __syncthreads();
    { f32x4 c4 = gemm_std_f16(Lk, Lk, lane); __syncthreads(); store_acc_16x16(Lk, c4, lane); } // Lk=L^4
    __syncthreads();
    // INV += INV @ L^4
    { f32x4 c = gemm_std_f16(INV, Lk, lane); __syncthreads();
      for (int i=0;i<4;i++){int m=(lane>>4)*4+i,n=lane&15; INV[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    // Lk = L^8 : rebuild L^2 -> L^4 -> L^8
    { f32x4 c2 = gemm_std_f16(Lm, Lm, lane); store_acc_16x16(Lk, c2, lane); }
    __syncthreads();
    { f32x4 c4 = gemm_std_f16(Lk, Lk, lane); __syncthreads(); store_acc_16x16(Lk, c4, lane); }
    __syncthreads();
    { f32x4 c8 = gemm_std_f16(Lk, Lk, lane); __syncthreads(); store_acc_16x16(Lk, c8, lane); }
    __syncthreads();
    // INV += INV @ L^8
    { f32x4 c = gemm_std_f16(INV, Lk, lane); __syncthreads();
      for (int i=0;i<4;i++){int m=(lane>>4)*4+i,n=lane&15; INV[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();

    // ---- write workspace ----
    for (int idx = lane; idx < C * D; idx += 64) {
        ws_kd[ht*C*D + idx] = kd[idx];
        ws_qd[ht*C*D + idx] = qd[idx];
        ws_kr[ht*C*D + idx] = kr[idx];
    }
    for (int d = lane; d < D; d += 64) ws_gt[ht*D + d] = g_total[d];
    for (int idx = lane; idx < C * C; idx += 64) {
        ws_inv[ht*C*C + idx] = f32_to_bf16(f16_to_f32(INV[idx]));
        ws_mqk[ht*C*C + idx] = Mqk[idx];
    }
}

}  // namespace flashkda_hip
