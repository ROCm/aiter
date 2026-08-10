// Standalone C-input GDN K2 prototype definitions.
//
// This intentionally does not alter gdn_defs.h or the existing Opus launcher.
// It is a fixed dense gfx942-oriented ABI for experimenting with the algebra
// used by the RTP FlyDSL megakernel:
//   Vd = C @ (beta * (V - exp(g) * K @ H))
// where C is the per-chunk inverse (I + L)^-1.
#pragma once

#include <cstddef>

#include "opus_gdn/gdn_defs.h"  // bf16_t only; existing K2 ABI is untouched.

struct gdn_k2_c_kargs {
    // Token-major dense inputs.
    const void* __restrict__ ptr_q;       // [B, T, H, K]   bf16
    const void* __restrict__ ptr_k;       // [B, T, H, K]   bf16
    const void* __restrict__ ptr_v;       // [B, T, H, V]   bf16
    const void* __restrict__ ptr_c;       // [B, T, H, BT]  bf16, C=(I+L)^-1
    const void* __restrict__ ptr_beta;    // [B, T, H]      fp32
    const void* __restrict__ ptr_g;       // [B, T, H]      fp32 local cumulative gate, natural-log domain
    const void* __restrict__ ptr_h0;      // [B, H, V, K]   fp32 (nullable)
    void* __restrict__ ptr_o;             // [B, T, H, V]   bf16
    void* __restrict__ ptr_ht;            // [B, H, V, K]   fp32 (nullable)
    int B;
    int T;
    int H;
    int K;
    int V;
    int NT;
    float scale;
    // Split-scan outputs.  They are appended so the fused ABI offsets above
    // stay unchanged; fused variants leave both pointers null.
    void* __restrict__ ptr_h_snap;         // [B, NT, H, V, K] bf16
    void* __restrict__ ptr_v_new;          // [B, T, H, V]     bf16
};
// This prototype is deliberately fixed to the requested launch geometry.  The
// trait is still a template so a future standalone launcher can instantiate a
// different V tile without changing the kernel body.
template<int BT_ = 64,
         int K_ = 128,
         int V_ = 128,
         int BV_ = 64,
         int NUM_WARPS_ = 4,
         bool PERSIST_K_ = false,
         bool CACHE_GATES_ = false,
         bool PERSIST_Q_ = true,
         int MIN_BLOCKS_ = 1,
         bool PREFETCH_Q_ = false,
         bool RELAX_BARRIERS_ = false,
         bool VECTOR_C_ = false,
         bool RETAIN_LAST_K_ = false,
         bool DIRECT_AV_ = false,
         bool WAVE_OWNED_ = false,
         bool FUSE_VD_K0_ = false,
         bool PREFETCH_D_K0_ = false,
         bool UNROLL_D_PACKS_ = false,
         int PREFETCH_D_K0_PACKS_ = 0,
         bool SPLIT_SCAN_ = false>
struct gdn_k2_c_traits {
    static constexpr int BT = BT_;
    static constexpr int K = K_;
    static constexpr int V = V_;
    static constexpr int BV = BV_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool PERSIST_K = PERSIST_K_;
    static constexpr bool CACHE_GATES = CACHE_GATES_;
    static constexpr bool PERSIST_Q = PERSIST_Q_;
    static constexpr int MIN_BLOCKS = MIN_BLOCKS_;
    static constexpr bool PREFETCH_Q = PREFETCH_Q_;
    static constexpr bool RELAX_BARRIERS = RELAX_BARRIERS_;
    static constexpr bool VECTOR_C = VECTOR_C_;
    static constexpr bool RETAIN_LAST_K = RETAIN_LAST_K_;
    static constexpr bool DIRECT_AV = DIRECT_AV_;
    static constexpr bool WAVE_OWNED = WAVE_OWNED_;
    static constexpr bool FUSE_VD_K0 = FUSE_VD_K0_;
    static constexpr bool PREFETCH_D_K0 = PREFETCH_D_K0_;
    static constexpr bool UNROLL_D_PACKS = UNROLL_D_PACKS_;
    static constexpr bool SPLIT_SCAN = SPLIT_SCAN_;
    static constexpr int PREFETCH_D_K0_PACKS = PREFETCH_D_K0
        ? 4
        : PREFETCH_D_K0_PACKS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int BK_SUB = 64;
    static constexpr int N_K = K / BK_SUB;
    static constexpr int SMEM_PAD = 4;

    using D_ATTN = bf16_t;
    using D_ACC = float;

    // Relaxed variants let a wave leave the attention phase without a CTA
    // barrier.  Double-buffer raw g so an early wave can stage the next chunk
    // without clobbering gates still consumed by a lagging wave.
    static constexpr int smem_g_bytes =
        (RELAX_BARRIERS ? 2 : 1) * BT * sizeof(D_ACC);
    static constexpr int smem_beta_bytes = BT * sizeof(D_ACC);
    static constexpr int smem_gate_cache_bytes = CACHE_GATES
        ? 2 * BT * sizeof(D_ACC)
        : 0;
    static constexpr int smem_vt_bytes = BV * (BT + SMEM_PAD) * sizeof(D_ATTN);
    static constexpr int smem_q_bytes = PERSIST_Q
        ? N_K * BT * (BK_SUB + SMEM_PAD) * sizeof(D_ATTN)
        : 0;
    static constexpr int smem_k_bytes = PERSIST_K
        ? N_K * BT * (BK_SUB + SMEM_PAD) * sizeof(D_ATTN)
        : 0;

    // The same phase pool is reused as:
    //   b/c: h^T [BV, BK+pad] + K [BT, BK+pad]
    //   C:   C [BT, BT+pad]
    //   d:   K_gated^T [BK, BT+pad]
    //   e:   K [BT, BK+pad], then gated QK^T [BT, BT+pad]
    static constexpr int phase_bc_bytes =
        (BV + BT) * (BK_SUB + SMEM_PAD) * sizeof(D_ATTN);
    static constexpr int phase_c_bytes = BT * (BT + SMEM_PAD) * sizeof(D_ATTN);
    static constexpr int phase_d_bytes = BK_SUB * (BT + SMEM_PAD) * sizeof(D_ATTN);
    static constexpr int pool_bytes =
        phase_bc_bytes > phase_c_bytes
            ? (phase_bc_bytes > phase_d_bytes ? phase_bc_bytes : phase_d_bytes)
            : (phase_c_bytes > phase_d_bytes ? phase_c_bytes : phase_d_bytes);

    static constexpr size_t smem_size_bytes() {
        return smem_g_bytes + smem_beta_bytes + smem_gate_cache_bytes
             + smem_vt_bytes
             + smem_q_bytes + smem_k_bytes + pool_bytes;
    }
};
