// C-split K1 preparation split into a 4-wave elementwise front-end and a
// compact one-wave triangular solve.  This removes the 31 KiB/one-wave
// occupancy bottleneck of the original all-in-one BT16 kernel on gfx942.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

namespace k1_split_prep_detail {

// CDNA3 implements a width-16 XOR tree without routing values through LDS.
// The row_ror:12 step is intentional: after XOR-8, destinations 0..3 read
// lanes 4..7, preserving the exact operand order of the original lane-0
// XOR-4 DAG.  Only lane 0 survives the reduction and is then broadcast to
// its 16-lane row with row_newbcast:0.
template <bool USE_DPP>
__device__ __forceinline__ float reduce_sum_16(float value) {
#if defined(__gfx942__)
    if constexpr (USE_DPP) {
        // DPP is fixed on src0, so the instruction evaluates remote + local.
        // q/k are finite in the inference contract, so their squared partial
        // sums are non-negative (and may overflow to infinity).  Over that
        // domain this is bit-identical to the original local + remote tree
        // while removing four standalone v_mov_dpp instructions.  Distinct
        // NaN payloads are intentionally outside that inference contract.
        float next;
        asm volatile(
            "v_add_f32 %0, %1, %2 row_ror:8 "
            "row_mask:0xf bank_mask:0xf"
            : "=v"(next) : "v"(value), "v"(value));
        value = next;
        asm volatile(
            "v_add_f32 %0, %1, %2 row_ror:12 "
            "row_mask:0xf bank_mask:0xf"
            : "=v"(next) : "v"(value), "v"(value));
        value = next;
        asm volatile(
            "v_add_f32 %0, %1, %2 quad_perm:[2,3,0,1] "
            "row_mask:0xf bank_mask:0xf"
            : "=v"(next) : "v"(value), "v"(value));
        value = next;
        asm volatile(
            "v_add_f32 %0, %1, %2 quad_perm:[1,0,3,2] "
            "row_mask:0xf bank_mask:0xf"
            : "=v"(next) : "v"(value), "v"(value));
        return next;
    }
#endif
#pragma unroll
    for (int offset = 8; offset >= 1; offset >>= 1)
        value += __shfl_xor(value, offset, 16);
    return value;
}

template <bool USE_DPP>
__device__ __forceinline__ float broadcast_row_lane0(float value) {
#if defined(__gfx942__)
    if constexpr (USE_DPP) {
        const int result = __builtin_amdgcn_mov_dpp(
            __builtin_bit_cast(int, value), 0x150, 0xf, 0xf, false);
        return __builtin_bit_cast(float, result);
    }
#endif
    return __shfl(value, 0, 16);
}

}  // namespace k1_split_prep_detail

template <bool VL, bool USE_DPP = false, bool PREP_BETA = false>
__global__ void __launch_bounds__(256)
k1_kda_split_prep_kernel(
        const __bf16* __restrict__ q_g,
        const __bf16* __restrict__ k_g,
        const __bf16* __restrict__ g_g,
        const float* __restrict__ A_log,
        const float* __restrict__ dt_bias,
        float scale, float gate_scale, int T_seq, int H,
        __bf16* __restrict__ ws_kd,
        __bf16* __restrict__ ws_qd,
        __bf16* __restrict__ ws_kr,
        float* __restrict__ ws_gt,
        __bf16* __restrict__ tmp_kinv,
        float* __restrict__ ws_decay,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int N, int total_tiles,
        const float* __restrict__ beta_g,
        float* __restrict__ cs_beta,
        const int* __restrict__ segment_prefix,
        int total_segments) {
    constexpr int C = 16, D = 128;
    const int tid = threadIdx.x;
    const int row_lane = tid & 15;
    int h, ht, t0, alen;
    int64_t beta_base = 0;
    int beta_alen = 0;
    bool produce_beta = false;
    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
        }
        const int local = gti - tile_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        const int seq_tiles = (len + C - 1) / C;
        if (local >= seq_tiles) return;
        ht = h * total_tiles + gti;
        t0 = int(bos) + local * C;
        alen = min(C, len - local * C);
        if constexpr (PREP_BETA) {
            produce_beta = (local & 3) == 0;
            if (produce_beta) {
                const int xs = h * total_segments +
                    segment_prefix[lo] + local / 4;
                beta_base = int64_t(xs) * 64;
                beta_alen = min(64, len - local * C);
            }
        }
    } else {
        const int nt = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht = bh * gridDim.x + nt;
        t0 = b * T_seq + nt * C;
        alen = min(C, T_seq - nt * C);
        if constexpr (PREP_BETA) {
            produce_beta = (nt & 3) == 0;
            if (produce_beta) {
                const int nseg = (int(gridDim.x) + 3) / 4;
                const int xs = bh * nseg + nt / 4;
                beta_base = int64_t(xs) * 64;
                beta_alen = min(64, T_seq - nt * C);
            }
        }
    }

    // q/k stay in registers through the final transform.  LDS carries the
    // gate prefixes; after the cumsum, row 15 is repurposed for total decay.
    struct alignas(16) PrepShared {
        float gc[C * D];
    };
    static_assert(sizeof(PrepShared) == C * D * sizeof(float),
                  "K1 split preparation LDS layout changed unexpectedly");
    static_assert(sizeof(PrepShared) == 8192,
                  "K1 split preparation must use exactly 8 KiB of LDS");
    __shared__ PrepShared smem;
    float* const gc = smem.gc;

    // One thread owns one aligned d8 vector in one row.  The old mapping made
    // every lane move eight scalar values separated by D; this mapping keeps
    // q/k/g traffic and gc publication on aligned 128-bit vector accesses.
    const int vec_m = tid >> 4;
    const int vec_d0 = row_lane * 8;
    const int vec_idx = vec_m * D + vec_d0;
    const float a = ex2(A_log[h] * KDA_LOG2E);
    bf16x8 qv{}, kv{}, gv{};
    f32x4 gcv0{}, gcv1{};
    if (vec_m < alen) {
        const int64_t off =
            (int64_t(t0 + vec_m) * H + h) * D + vec_d0;
        qv = *reinterpret_cast<const bf16x8*>(q_g + off);
        kv = *reinterpret_cast<const bf16x8*>(k_g + off);
        gv = *reinterpret_cast<const bf16x8*>(g_g + off);
        const f32x4 db0 =
            *reinterpret_cast<const f32x4*>(dt_bias + h * D + vec_d0);
        const f32x4 db1 =
            *reinterpret_cast<const f32x4*>(dt_bias + h * D + vec_d0 + 4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            gcv0[i] = gate_scale * sigmoid_tanh(
                a * (bf16_to_f32(gv[i]) + db0[i]));
            gcv1[i] = gate_scale * sigmoid_tanh(
                a * (bf16_to_f32(gv[i + 4]) + db1[i]));
        }
    }
    *reinterpret_cast<f32x4*>(gc + vec_idx) = gcv0;
    *reinterpret_cast<f32x4*>(gc + vec_idx + 4) = gcv1;

    // The 16 threads that own a row first sum their eight register-resident
    // q/k elements, then reduce strictly within that 16-lane partition.  A
    // width of 16 makes shuffle lane 0 relative to each partition, so neither
    // the xor tree nor the final broadcast can cross a row boundary.
    float qs = 0.0f, ks = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float qf = bf16_to_f32(qv[i]);
        const float kf = bf16_to_f32(kv[i]);
        qs += qf * qf;
        ks += kf * kf;
    }
    qs = k1_split_prep_detail::reduce_sum_16<USE_DPP>(qs);
    ks = k1_split_prep_detail::reduce_sum_16<USE_DPP>(ks);
    float qinv_row = 0.0f, kinv_row = 0.0f;
    if (row_lane == 0) {
        qinv_row = rsqrtf(qs + 1e-6f);
        kinv_row = rsqrtf(ks + 1e-6f);
    }
    qinv_row =
        k1_split_prep_detail::broadcast_row_lane0<USE_DPP>(qinv_row);
    kinv_row =
        k1_split_prep_detail::broadcast_row_lane0<USE_DPP>(kinv_row);

    // Publish every row's gate values before the column-wise cumsums begin.
    __syncthreads();

    // P1 waves 2/3 are otherwise idle while waves 0/1 perform the 128 gate
    // prefix scans.  The first tile in each BT64 segment uses all of wave 3 to
    // publish its complete activated-beta row, matching P2's original mapping
    // with one fully utilized sigmoid wave instead of four quarter waves.
    if constexpr (PREP_BETA) {
        if (produce_beta && tid >= 192) {
            const int row = tid & 63;
            cs_beta[int64_t(beta_base) + row] = row < beta_alen
                ? sigmoid_tanh(beta_g[int64_t(t0 + row) * H + h])
                : 0.0f;
        }
    }

    // One thread per K column performs the exact sequential chunk cumsum.
    if (tid < D) {
        float acc = 0.0f;
        #pragma unroll
        for (int m = 0; m < C - 1; ++m) {
            acc += gc[m * D + tid];
            gc[m * D + tid] = acc;
        }
        // Each thread exclusively owns one column.  Finish its final cumsum
        // add before replacing row 15 with the representation consumed after
        // the block-wide publication barrier below.
        acc += gc[(C - 1) * D + tid];
        const float decay = ex2(acc);
        gc[(C - 1) * D + tid] = decay;
        ws_gt[int64_t(ht) * D + tid] = acc;
        // The ws_mqk tile is exactly 128 fp32 values.  C-split preparation
        // temporarily publishes total decay there; legacy split_solve may
        // overwrite it later, while the RTP-K6 route consumes it directly.
        ws_decay[int64_t(ht) * D + tid] = decay;
    }
    __syncthreads();

    const __bf16 scale_bf = f32_to_bf16(scale);
    const f32x4 gc0 =
        *reinterpret_cast<const f32x4*>(gc + vec_idx);
    const f32x4 gc1 =
        *reinterpret_cast<const f32x4*>(gc + vec_idx + 4);
    const int decay_idx = (C - 1) * D + vec_d0;
    const f32x4 decay0 =
        *reinterpret_cast<const f32x4*>(gc + decay_idx);
    const f32x4 decay1 =
        *reinterpret_cast<const f32x4*>(gc + decay_idx + 4);
    bf16x8 kd_v{}, ki_v{}, kr_v{}, qd_v{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float gc_i = i < 4 ? gc0[i] : gc1[i - 4];
        const float decay_i = i < 4 ? decay0[i] : decay1[i - 4];
        // The supported gate keeps a full BT16 prefix within the normal fp32
        // range (worst case is about 2^-115).  Form every prefix decay before
        // selecting row 15's already exponentiated value, keeping the select
        // branch-free while preserving row 15's exact output.  Form the
        // inverse in fp32 and round both operands only after the reciprocal.
        const float dp_prefix = ex2(gc_i);
        const float dp_f = vec_m == C - 1 ? decay_i : dp_prefix;
        const float dn_f = __builtin_amdgcn_rcpf(dp_f);
        const __bf16 dp = f32_to_bf16(dp_f);
        const __bf16 dn = f32_to_bf16(dn_f);
        const __bf16 dt = f32_to_bf16(decay_i);
        const float kn = bf16_to_f32(kv[i]) * kinv_row;
        const float qn = bf16_to_f32(qv[i]) * qinv_row;
        kd_v[i] = f32_to_bf16(kn * bf16_to_f32(dp));
        ki_v[i] = f32_to_bf16(kn * bf16_to_f32(dn));
        const __bf16 qt = f32_to_bf16(qn * bf16_to_f32(dp));
        kr_v[i] =
            f32_to_bf16(bf16_to_f32(ki_v[i]) * bf16_to_f32(dt));
        qd_v[i] =
            f32_to_bf16(bf16_to_f32(qt) * bf16_to_f32(scale_bf));
    }
    const int64_t ws_vec_off = int64_t(ht) * C * D + vec_idx;
    *reinterpret_cast<bf16x8*>(ws_kd + ws_vec_off) = kd_v;
    *reinterpret_cast<bf16x8*>(tmp_kinv + ws_vec_off) = ki_v;
    *reinterpret_cast<bf16x8*>(ws_kr + ws_vec_off) = kr_v;
    *reinterpret_cast<bf16x8*>(ws_qd + ws_vec_off) = qd_v;
}

template <bool VL>
__global__ void __launch_bounds__(64)
k1_kda_split_solve_kernel(
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ tmp_kinv,
        __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ ws_mqk,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int N, int total_tiles, int T_seq, int H) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    int h, ht, t0, alen;
    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
        }
        const int local = gti - tile_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        if (local >= (len + C - 1) / C) return;
        ht = h * total_tiles + gti;
        t0 = int(bos) + local * C;
        alen = min(C, len - local * C);
    } else {
        const int nt = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht = bh * gridDim.x + nt;
        t0 = b * T_seq + nt * C;
        alen = min(C, T_seq - nt * C);
    }

    __shared__ __bf16 kd[C * D], qd[C * D], ki[C * D], mqk[C * C];
    __shared__ float beta[C];
    __shared__ _Float16 lm[C * C], inv[C * C], lk[C * C];
    copy_bf16_vec(kd, ws_kd + int64_t(ht) * C * D, C * D, lane);
    copy_bf16_vec(qd, ws_qd + int64_t(ht) * C * D, C * D, lane);
    copy_bf16_vec(ki, tmp_kinv + int64_t(ht) * C * D, C * D, lane);
    if (lane < C)
        beta[lane] = lane < alen
            ? sigmoid_tanh(beta_g[int64_t(t0 + lane) * H + h]) : 0.0f;
    __syncthreads();

    f32x4 cl = gemm_contract_last<__bf16, D>(kd, ki, lane);
    f32x4 cm = gemm_contract_last<__bf16, D>(qd, ki, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        lm[m * C + n] = m > n
            ? f32_to_f16(cl[i]) * f32_to_f16(beta[m]) : (_Float16)0.0f;
        mqk[m * C + n] = m >= n ? f32_to_bf16(cm[i]) : (__bf16)0.0f;
        inv[m * C + n] = (_Float16)(m == n ? 1.0f : 0.0f) - lm[m * C + n];
    }
    __syncthreads();

    { f32x4 c = gemm_std_f16(lm, lm, lane); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(inv, lk, lane); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    { f32x4 c = gemm_std_f16(lm, lm, lane); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(lk, lk, lane); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(inv, lk, lane); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    { f32x4 c = gemm_std_f16(lm, lm, lane); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(lk, lk, lane); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(lk, lk, lane); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = gemm_std_f16(inv, lk, lane); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();

    for (int idx = lane; idx < C * C; idx += 64) {
        ws_inv[int64_t(ht) * C * C + idx] = f32_to_bf16(f16_to_f32(inv[idx]));
        ws_mqk[int64_t(ht) * C * C + idx] = mqk[idx];
    }
}

}  // namespace flashkda_hip
