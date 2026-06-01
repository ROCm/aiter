// Shared types and constants between device kernel and host code
#pragma once

using bf16_t = __bf16;
using fp16_t = __fp16;

// Kernel arguments for MLA persistent decode.
//
// The pointer layout below mirrors the existing hk_mla decode launchers (see
// `csrc/kernels/mla/hk/mi35x_v32_fwd_decode_*.cuh`) so the public host API
// (`hk_mla_a16w16_16mx8_32nx1_ps`) is interchangeable: callers pass the same
// torch::Tensors as for `hk_mla_decode_fwd`, and the launcher fills these
// fields directly. No 64-bit pointer-of-pointers indirection.
struct mla_decode_ps_kargs {
    void* __restrict__ split_out_ptr;              // [split_total_len, H, D_v]
    void* __restrict__ split_lse_ptr;              // [split_total_len, H]
    void* __restrict__ final_out_ptr;              // [total_qlen, H, D_v]
    void* __restrict__ final_lse_ptr;              // [total_qlen, H] (optional)
    const void* __restrict__ q_ptr;                // [total_qlen, H, D_qk]
    const void* __restrict__ kv_ptr;               // [total_tokens, 1, D_kv]
    const int*  __restrict__ q_indptr;             // [N+1]
    const int*  __restrict__ work_indptr;          // [num_tgs + 1]
    const int*  __restrict__ work_info_set;        // [num_works * 8] dwords
    const int*  __restrict__ kv_indices;           // [total_kv_index_len]
    const int*  __restrict__ kv_last_page_lens;    // [N] (optional, for paged KV)
    float        scalar;                           // softmax_scale
    unsigned int s_gqa_ratio;
    // Q-side stride (in elements). Q is [total_qlen, H, D_qk].
    unsigned int stride_qo_n;                      // = H * D_qk
    unsigned int stride_qo_h;                      // = D_qk
    // Output-side strides (in elements). final/split output is [..., H, D_v];
    // D_v != D_qk for MLA so they MUST be tracked separately.
    unsigned int stride_o_n;                       // per-row stride of final / split out
    unsigned int stride_o_h;                       // per-head stride of final / split out (= D_v)
    unsigned int s_Bs;                             // KV row stride in bytes
    unsigned int s_log2_plen;                      // log2(page_size)
    unsigned int kv_buffer_bytes;                  // total bytes of kv_ptr buffer (for buffer-rsrc OOB)
    void* __restrict__ q_scale_ptr;                // [total_qlen]    (optional, fp8 path)
    void* __restrict__ kv_scale_ptr;               // [total_tokens]  (optional, fp8 path)
};

// Configuration traits for MLA decode persistent kernel.
//
// MLA shape conventions:
//   - K (and Q) per-head dim = D_TILE_SIZE   (default 576 = 512 kv_lora + 64 rope)
//   - V          per-head dim = V_D_TILE_SIZE (default 512 = kv_lora_rank only)
//   - The KV cache is a single buffer with D_TILE_SIZE columns per token; V is
//     the first V_D_TILE_SIZE columns of every K row (rope tail is ignored
//     during PV).
template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 32,
         int D_TILE_SIZE_ = 576,
         int V_D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 8,
         typename D_ATTN_ = bf16_t,
         bool CAUSAL_ = false>
struct mla_decode_ps_traits {
    static constexpr int Q_TILE_SIZE   = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE  = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE   = D_TILE_SIZE_;   // QK feature dim
    static constexpr int V_D_TILE_SIZE = V_D_TILE_SIZE_; // PV feature dim
    static constexpr int NUM_WARPS     = NUM_WARPS_;
    // Apply causal masking. For MLA decode the M tile packs (q_token, q_head)
    // with head as the fast axis, so a query token attends to keys up to its
    // own position. Only the KV slice reaching the sequence end (kv_offset==0)
    // carries the diagonal; see attn_mask_causal_tile / the call sites.
    static constexpr bool CAUSAL = CAUSAL_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // D slicing: SLICE_D=32 chunks. K side has NUM_K_SLICES iterations,
    // V side only iterates the leading V_D_TILE_SIZE columns.
    static constexpr int SLICE_D       = 32;
    static constexpr int NUM_K_SLICES  = D_TILE_SIZE   / SLICE_D;   // 18 for 576
    static constexpr int NUM_V_SLICES  = V_D_TILE_SIZE / SLICE_D;   // 16 for 512
    static_assert(D_TILE_SIZE   % SLICE_D == 0);
    static_assert(V_D_TILE_SIZE % SLICE_D == 0);
    static_assert(V_D_TILE_SIZE <= D_TILE_SIZE,
                  "V dim must fit inside K dim (V is a prefix of the KV row)");

    // GEMM0: S[Q_TILE x KV_TILE] = Q[Q_TILE x SLICE_D] @ K^T[SLICE_D x KV_TILE]
    static constexpr int GEMM0_E_M = Q_TILE_SIZE  / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = SLICE_D      / W_K;

    // GEMM1: O[Q_TILE x SLICE_D] = P[Q_TILE x KV_TILE] @ V[KV_TILE x SLICE_D]
    static constexpr int GEMM1_E_M = Q_TILE_SIZE  / W_M;
    static constexpr int GEMM1_E_N = SLICE_D      / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K;

    static constexpr int VEC_Q    = 8;
    static constexpr int VEC_KV   = 8;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    static constexpr int D_128B_SIZE        = 128 / sizeof(D_ATTN);
    static_assert(VEC_KV == 16 / sizeof(D_ATTN));
    static constexpr int smem_linear_wave   = WARP_SIZE * 16 / sizeof(D_ATTN); // 64*16/2 = 512
    static constexpr int smem_n_per_wave    = smem_linear_wave / D_128B_SIZE;  // 512/64 = 8
    static constexpr int smem_n_rpt         = KV_TILE_SIZE / smem_n_per_wave;  // 32/8  = 4
    static constexpr int smem_d_rpt         = D_TILE_SIZE   / D_128B_SIZE;     // K side full row -> 9 (D=576)
    static constexpr int smem_v_d_rpt       = V_D_TILE_SIZE / D_128B_SIZE;     // V side prefix   -> 8 (V=512)
    static constexpr int smem_d_rpt_tail    = smem_d_rpt - smem_v_d_rpt;       // rope tail chunks -> 1 (D-V=64)
    static constexpr int smem_padding_32B   = 32 / sizeof(D_ATTN);
    static constexpr int smem_kv_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_padding_32B);

    // ----- KV gmem->smem two-stage load -----
    // The gmem->smem KV load partitions the D-axis across `warps_d` warp groups,
    // and each thread issues `chunks_per_warp_d` chunk loads. For D=576 with
    // NUM_WARPS=8 -> warps_d=2, 9/2=4.5 doesn't divide cleanly.  We split the
    // load into two stages so every stage divides cleanly:
    //
    //   stage 1 (main):  KV_TILE x V_D_TILE_SIZE  (32x512) -- 8 chunks, 8 warps,
    //                    chunks_per_warp_d = smem_v_d_rpt / warps_d = 8/2 = 4 ?
    //                    every warp/thread participates.
    //
    //   stage 2 (tail):  KV_TILE x (D-V)         (32x64)  -- 1 chunk, only the
    //                    first `smem_n_rpt` warps (warps 0..smem_n_rpt-1) issue
    //                    the load; warps `[smem_n_rpt, NUM_WARPS)` skip the
    //                    tail entirely. Each active thread issues 1 vec-load
    //                    (active threads = smem_n_rpt x WARP_SIZE = 4x64 = 256;
    //                     vec-loads needed = 32 rows x 64 dim / 8 vec = 256 ?).
    static constexpr int warps_d_for_load   = NUM_WARPS / smem_n_rpt;
    static_assert(warps_d_for_load * smem_n_rpt == NUM_WARPS,
                  "NUM_WARPS must be a multiple of smem_n_rpt");
    static_assert(smem_v_d_rpt % warps_d_for_load == 0,
                  "V_D_TILE_SIZE/D_128B_SIZE must be divisible by warps_d (NUM_WARPS/smem_n_rpt) "
                  "for the main-stage KV load to divide cleanly");
    static_assert((D_TILE_SIZE - V_D_TILE_SIZE) == D_128B_SIZE,
                  "rope tail must be exactly one 128B D-chunk");

    static constexpr int kv_buffer_load_insts_main =
        (KV_TILE_SIZE * V_D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV);
    static constexpr int kv_buffer_load_insts_tail =
        (KV_TILE_SIZE * (D_TILE_SIZE - V_D_TILE_SIZE)) / (smem_n_rpt * WARP_SIZE * VEC_KV);
    // Per-active-thread total issue count = main + tail (used by waitcnt math).
    // Warps in [smem_n_rpt, NUM_WARPS) only issue `kv_buffer_load_insts_main`,
    // but vmcnt is per-thread and tracks each thread's own issued loads.
    static constexpr int kv_buffer_load_insts = kv_buffer_load_insts_main + kv_buffer_load_insts_tail;

    static constexpr int k_ds_read_insts      = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_read_insts      = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V);

    static constexpr size_t smem_size_bytes() {
        return 4 * smem_kv_tile_elems * sizeof(D_ATTN);
    }
};

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
