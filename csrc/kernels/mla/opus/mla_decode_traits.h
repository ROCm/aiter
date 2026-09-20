#pragma once

#include <algorithm>

#include <opus/dtypes.hpp>

#include "mla_decode_kargs.h"

using bf16_t = opus::dtypes::bf16;
using fp16_t = opus::dtypes::fp16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 8,
          typename D_NOPE_  = fp8_t,
          typename D_ROPE_  = bf16_t,
          typename D_OUT_   = bf16_t>
struct opus_mla_decode_mxfp8_16mx8_32nx1_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE         = 512;
    static constexpr int D_ROPE_SIZE         = 64;
    static constexpr int D_HEAD_SIZE         = D_NOPE_SIZE + D_ROPE_SIZE;
    static constexpr int D_SCALE_SIZE        = D_NOPE_SIZE / 32;
    static constexpr int D_SCALE_PADDED_SIZE = 32;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M      = 16;
    static constexpr int W_N      = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_NOPE_SIZE / SLICE_D;

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = SLICE_D / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE;

    static constexpr int VEC_Q_NOPE  = 16;
    static constexpr int VEC_Q_ROPE  = 8;
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8;
    static constexpr int VEC_TR_V    = 4;
    static constexpr int VEC_O       = 4;

    static constexpr int D_128B_NOPE_SIZE      = 128 / sizeof(D_NOPE);
    static constexpr int dwordx4_size          = 16;
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_NOPE);
    static constexpr int smem_n_per_wave       = 8;
    static constexpr int smem_n_rpt            = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt_nope       = D_NOPE_SIZE / D_128B_NOPE_SIZE;
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_NOPE);
    static constexpr size_t smem_k_nope_bytes  = smem_n_rpt * smem_d_rpt_nope *
                                                (smem_linear_wave_nope + smem_padding_32B_nope) *
                                                sizeof(D_NOPE);

    static constexpr int D_128B_ROPE_SIZE      = 128 / sizeof(D_ROPE);
    static constexpr int smem_linear_wave_rope = WARP_SIZE * dwordx4_size / sizeof(D_ROPE);
    static constexpr int smem_d_rpt_rope       = D_ROPE_SIZE / D_128B_ROPE_SIZE;
    static constexpr int smem_padding_32B_rope = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_k_rope_bytes  = smem_n_rpt * smem_d_rpt_rope *
                                                (smem_linear_wave_rope + smem_padding_32B_rope) *
                                                sizeof(D_ROPE);

    static constexpr int smem_v_padding = 32 / sizeof(D_ROPE);
    static constexpr size_t smem_v_bytes =
        KV_TILE_SIZE * (D_NOPE_SIZE + smem_v_padding) * sizeof(D_ROPE);

    static constexpr int smem_mxscl_padding = 4 / sizeof(D_NOPE);
    static constexpr size_t smem_mxscl_bytes =
        smem_n_rpt * (D_SCALE_PADDED_SIZE * smem_n_per_wave + smem_mxscl_padding) * sizeof(D_NOPE);

    static constexpr size_t smem_kv_bytes()
    {
        return std::max(smem_k_nope_bytes + smem_k_rope_bytes, smem_v_bytes);
    }

    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE) // nope = 2
        + 1; // rope = 1 for warp_id < 4 or mxscl = 1 for warp_id >= 4
    static constexpr int k_nope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct opus_mla_decode_fp8_kargs
{
    const void* __restrict__ q_buffer_ptr;
    const void* __restrict__ q_scale_ptr;
    const void* __restrict__ kv_buffer_ptr;
    const void* __restrict__ kv_scale_ptr;
    void* __restrict__ o_accum;
    void* __restrict__ lse_accum;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;
    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;

    // Per-batch length of the last KV page, in tokens (1..page_size). Read only when
    // PAGE_SIZE > 1 and the work item covers the batch tail (kv_offset == 0), where that
    // page is partial and the page range in work_info would overstate the token count.
    // May be null when PAGE_SIZE == 1.
    const int* __restrict__ kv_last_page_lens;

    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    // Elements between consecutive token rows of the KV cache. Deliberately NOT the dim-0
    // stride once the cache is paged: a [num_page, page_size, 1, D] tensor strides
    // page_size * D on dim 0, and the kernel steps whole pages by scaling this by
    // PAGE_SIZE itself.
    int stride_kv_page;
    float softmax_scale;
};

// ============================================================================
// Traits for the *combined* fp8 MLA decode kernel (mla_decode_fwd_16mx8_32nx1
// mla_decode_fp8_16mx8_32nx1.hpp).
//
// Differences vs. opus_mla_decode_mxfp8_16mx8_32nx1_traits above:
//   1. q_buffer / kv_buffer are a *single* contiguous d = D_HEAD_SIZE = 576 fp8
//      tensor (row-major, d contiguous). The "nope" part is d in [0, 512) and
//      the "rope" part is d in [512, 576); both are read from the same base
//      pointer with row stride = D_HEAD_SIZE. rope is fp8 (not bf16).
//   2. No mxfp8 micro-scaling. q_scale_ptr / kv_scale_ptr are per-tensor scalar
//      descales (a single float each, cf. s_descale_q / s_descale_k in the SP3
//      kernel). QK^T scores are multiplied by descale_q * descale_k afterwards,
//      so there is no scale LDS region.
//   3. V (the nope part of KV) is NOT dequantized and NOT re-stored: it is
//      transpose-read (ds_read_b64_tr_b8) straight out of the fp8 K-nope LDS and
//      fed to a plain fp8 PV MFMA; descale_k is applied once, on the output.
//      This mirrors the SP3 kernel (v_cvt_pk_fp8_f32(S) + fp8 P*V).
// ============================================================================
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 8,
          typename D_Q_     = fp8_t,
          typename D_K_     = fp8_t,
          typename D_OUT_   = bf16_t,
          bool CAUSAL_      = false,
          bool LARGE_KV_    = false>
struct opus_mla_decode_fp8_16mx8_32nx1_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    // KV cache past the 4 GiB a buffer descriptor can address; see the KV load in
    // mla_decode_fp8_16mx8_32nx1.hpp. Costs ~1-2% and 3 spilled VGPR, so
    // the host only turns it on for the caches that need it.
    static constexpr bool LARGE_KV = LARGE_KV_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE; // 576, combined buffer row stride

    using D_Q   = D_Q_; // fp8 activations (Q)
    using D_K   = D_K_; // fp8 weights (K, and the source of V)
    using D_V   = D_K_; // V is raw fp8 (= K nope); PV is fp8*fp8, no dequant
    using D_OUT = D_OUT_;
    using D_ACC = float;

    // The waves tile M here (each owns Q_TILE_SIZE query rows and all of the output d), so
    // T_M is the wave count and T_N is 1. v_o is sized Q_TILE_SIZE * D_NOPE_SIZE / (T_N *
    // WARP_SIZE) and reinterpreted as NUM_D_SLICES slices, so swapping these two silently
    // undersizes it by 8x and the PV accumulation runs off the end (100% NaN at nhead 128).
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    // GEMM0 (QK^T): the nope part uses the gfx950 16x16x128 f8f6f4 MFMA with the
    // per-block E8M0 scale hard-set to 127 (== 2^0, i.e. no micro-scaling); the
    // per-tensor descale is applied to the scores instead. The rope part uses
    // the plain fp8 16x16x32 MFMA.
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;

    static constexpr int SLICE_D      = 32;
    static constexpr int NUM_D_SLICES = D_NOPE_SIZE / SLICE_D; // 16, GEMM1 output d-slices

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;      // 1
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N;     // 2
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE; // 4
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE; // 2

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;       // 1
    static constexpr int GEMM1_E_N = SLICE_D / W_N;           // 2
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_ROPE; // 1

    static constexpr int VEC_Q_NOPE  = 16; // fp8 dwordx4 global/ds vector
    static constexpr int VEC_Q_ROPE  = 8;  // fp8 rope
    static constexpr int VEC_KV_NOPE = 16;
    static constexpr int VEC_KV_ROPE = 8; // fp8 rope: read (MFMA B-operand) vector
    // rope ASYNC-LOAD vector: gfx950 buffer_load...lds has no 8-byte (dwordx2)
    // form, so rope must be loaded with a supported b32 (4-byte) transfer using
    // 16 lanes/token. An 8B (VEC_KV_ROPE) load emits no instruction -> NaN.
    static constexpr int VEC_KV_ROPE_LD = 4;
    static constexpr int VEC_TR_V       = 8;
    static constexpr int VEC_O          = 4;

    static constexpr int dwordx4_size = 16;

    // ----- K nope LDS geometry (fp8), identical scheme to the 3-buffer variant's nope -----
    static constexpr int D_128B_NOPE_SIZE      = 128 / sizeof(D_K);                      // 128
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_K); // 1024
    static constexpr int smem_n_per_wave       = 8;
    static constexpr int smem_n_rpt            = KV_TILE_SIZE / smem_n_per_wave; // 4
    static constexpr int smem_d_rpt_nope       = D_NOPE_SIZE / D_128B_NOPE_SIZE; // 4
    static constexpr int smem_padding_32B_nope = 32 / sizeof(D_K);               // 32
    static constexpr size_t smem_k_nope_bytes  = smem_n_rpt * smem_d_rpt_nope *
                                                (smem_linear_wave_nope + smem_padding_32B_nope) *
                                                sizeof(D_K);

    // ----- V transpose-read bank swizzle -----
    static constexpr int SWZ_D_BYTES = VEC_KV_NOPE;         // 16, one d-group
    static constexpr int SWZ_TOK_BIT = smem_n_per_wave / 2; // token-in-line bit 2

    // ----- K rope LDS geometry (fp8). rope d = 64 fits in a single sub-line
    //       so we treat one 64-wide "128B-like" chunk per row. -----
    static constexpr int D_128B_ROPE_SIZE      = D_ROPE_SIZE; // 64
    static constexpr int smem_linear_wave_rope = WARP_SIZE * VEC_KV_ROPE_LD / sizeof(D_K);
    static constexpr int smem_d_rpt_rope       = 1;
    static constexpr int smem_padding_rope     = 16 / sizeof(D_K); // 16
    // New rope layout: one warp per LDS line (4 tokens/line), NUM_WARPS lines
    // total. The read (make_layout_rk_rope) reaches lines 4..7 via the GEMM0_E_N
    // stride, and the store (make_layout_sk_rope) writes lines 0..7 via warp_id,
    // so the region must hold NUM_WARPS lines, not smem_n_rpt.
    static constexpr size_t smem_k_rope_bytes = NUM_WARPS * smem_d_rpt_rope *
                                                (smem_linear_wave_rope + smem_padding_rope) *
                                                sizeof(D_K); // 2176

    // V is NOT dequantized and NOT re-stored: it is transpose-read (fp8) straight
    // out of the K-nope LDS, so the per-slot KV footprint is just K (nope+rope).
    static constexpr size_t smem_kv_bytes() { return smem_k_nope_bytes + smem_k_rope_bytes; }

    // fp8 nope + fp8 rope: one dwordx4 (16 fp8) per thread each -> 2 + 1 loads.
    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_KV_NOPE) +
        (KV_TILE_SIZE * D_ROPE_SIZE) / (BLOCK_SIZE * VEC_KV_ROPE_LD); // 2 + 1 = 3
    static constexpr int k_nope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE);
    static constexpr int k_rope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_KV_ROPE);
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V);
};

// ============================================================================
// Traits for the 16mx1 / 16nx8 variant. The combined d = 576 fp8 buffer and the per-tensor
// scalar descale are as above; what changes is that the NUM_WARPS waves tile N instead of M.
//
// The Q_TILE_SIZE rows are ONE tile every wave shares, so Q cannot stay in registers and
// gets its own LDS region, while each wave owns KV_TILE_SIZE of the
// NUM_WARPS * KV_TILE_SIZE tokens one GEMM0 step consumes. GEMM1 contracts over all of
// them, so P round-trips LDS as well and each wave keeps D_NOPE_SIZE / NUM_WARPS of the
// output d.
// ============================================================================
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 16,
          int NUM_WARPS_    = 8,
          typename D_Q_     = fp8_t,
          typename D_K_     = fp8_t,
          typename D_OUT_   = bf16_t,
          bool CAUSAL_      = false,
          bool LARGE_KV_    = false,
          int PAGE_SIZE_    = 1>
struct opus_mla_decode_fp8_16mx1_16nx8_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    // Tokens per KV page. At 1, kv_indices is a per-token table; above that it is a block
    // table indexed by token / PAGE_SIZE and a page is PAGE_SIZE contiguous token rows.
    //
    // Worth having only because a 576-byte token row is 4.5 cache lines, so a row fetched
    // in isolation drags in 640 bytes -- measured 1.113x more DRAM traffic than the kernel
    // consumes. Two adjacent rows come to exactly 9 lines, which recovers it: a read-only
    // bandwidth probe on gfx950 measures 5.44 TB/s for scattered 576-byte rows against
    // 6.04 TB/s for 1152-byte pairs, out of a 6.29 TB/s coalesced ceiling. PAGE_SIZE 4
    // adds ~0.5% over 2, which is where this stops paying.
    //
    // Capped at 4 because the within-page token offset has to stay wave-uniform: the DMA
    // deals token (warp_id % 4) + 4k to warp_id, so token % PAGE_SIZE is a function of
    // warp_id alone up to 4 and becomes per-lane at 8.
    static constexpr int PAGE_SIZE = PAGE_SIZE_;
    static_assert(PAGE_SIZE == 1 || PAGE_SIZE == 2 || PAGE_SIZE == 4,
                  "PAGE_SIZE must be 1, 2 or 4");
    // KV cache past the 4 GiB a buffer descriptor can address; see the KV load in
    // mla_decode_fp8_16mx1_16nx8.hpp. Costs ~1-2% and 3 spilled VGPR, so
    // the host only turns it on for the caches that need it.
    static constexpr bool LARGE_KV = LARGE_KV_;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_NOPE_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE; // 576, combined buffer row stride
    static constexpr int D_HEAD_SIZE_PADDING = D_HEAD_SIZE + D_ROPE_SIZE; // 640

    using D_Q   = D_Q_; // fp8 activations (Q)
    using D_K   = D_K_; // fp8 weights (K, and the source of V)
    using D_V   = D_K_; // V is raw fp8 (= K nope); PV is fp8*fp8, no dequant
    using D_OUT = D_OUT_;
    using D_ACC = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int SLICE_D      = D_NOPE_SIZE / NUM_WARPS; // 64
    static constexpr int NUM_D_SLICES = SLICE_D / W_N; // 512 / 8 / 16 = 4, GEMM1 output d-slices

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;         // 1
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / W_N;        // 1
    static constexpr int GEMM0_E_K = D_HEAD_SIZE_PADDING / W_K; // 5

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;              // 1
    static constexpr int GEMM1_E_N = NUM_D_SLICES;                   // 4
    static constexpr int GEMM1_E_K = KV_TILE_SIZE * NUM_WARPS / W_K; // 1

    static constexpr int VEC_Q       = 16; // fp8 dwordx4 global/ds vector
    static constexpr int VEC_KV      = 16;
    static constexpr int VEC_READ_P  = 8; // one 8B half of a block row, i.e. a ds_read_b64
    static constexpr int VEC_WRITE_P = 4;
    static constexpr int VEC_TR_V    = 8;
    static constexpr int VEC_O       = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int dword_size   = 4;

    // ----- Q LDS geometry (fp8) -----
    // One block of smem_linear_wave_q + smem_padding_32B per wave, holding that wave's
    // buffer_load_lds verbatim: smem_n_rpt_q waves cover the Q_TILE_SIZE rows
    // (smem_n_per_wave_q rows each) and the remaining smem_d_rpt_q cover d
    // (smem_d_per_wave_q each). See make_layout_gq in the kernel.
    static constexpr int smem_linear_wave_q = WARP_SIZE * dwordx4_size / sizeof(D_Q); // 1024
    static constexpr int smem_n_per_wave_q  = 4;
    static constexpr int smem_n_rpt_q       = Q_TILE_SIZE / smem_n_per_wave_q;        // 4
    static constexpr int smem_d_per_wave_q  = smem_linear_wave_q / smem_n_per_wave_q; // 256
    static constexpr int smem_d_rpt_q       = D_NOPE_SIZE / smem_d_per_wave_q;        // 2
    static constexpr int smem_padding_32B   = 32 / sizeof(D_Q);
    static constexpr size_t smem_q_bytes    = smem_n_rpt_q * smem_d_rpt_q *
                                           (smem_linear_wave_q + smem_padding_32B) *
                                           sizeof(D_Q); // 8448B
    static_assert(smem_n_rpt_q * smem_d_rpt_q == NUM_WARPS, "one Q LDS block per wave");

    // ----- K LDS geometry (fp8) -----
    static constexpr int smem_linear_wave_kv = WARP_SIZE * dwordx4_size / sizeof(D_K); // 1024
    static constexpr int smem_n_per_wave_kv  = 16;
    static constexpr int smem_n_rpt_kv       = KV_TILE_SIZE * NUM_WARPS / smem_n_per_wave_kv; // 8
    static constexpr int smem_d_per_wave_kv  = smem_linear_wave_kv / smem_n_per_wave_kv;      // 64
    static constexpr int smem_d_rpt_kv       = D_HEAD_SIZE / smem_d_per_wave_kv;              // 9
    static constexpr size_t smem_kv_bytes =
        smem_n_rpt_kv * smem_d_rpt_kv * (smem_linear_wave_kv + smem_padding_32B) * sizeof(D_K);

    // ----- P LDS geometry (fp8) -----
    // A block carries smem_linear_wave_p bytes of scores but sits on a wider pitch, and its rows
    // are permuted and half-swapped: together those three take both the write and the read to
    // zero bank conflicts, see make_layout_sp. The read forces the pitch to 128 mod 256, and the
    // cheapest such pitch leaves a third of a block spare.
    static constexpr int smem_linear_wave_p = WARP_SIZE * dword_size / sizeof(D_V); // 256
    static constexpr int P_SWZ_ROW_MUL      = 11; // rows land on 16 * (11 * r % W_M)
    static constexpr int smem_p_pitch       = 384;
    static constexpr size_t smem_p_bytes    = NUM_WARPS * smem_p_pitch * sizeof(D_V); // 3072B
    static_assert(smem_p_pitch % 256 == 128 && smem_p_pitch >= smem_linear_wave_p,
                  "the two blocks a read phase spans must sit 16 bank slots apart");

    static constexpr size_t smem_q_padding_bytes = 9 * 1024 * sizeof(D_Q);

    // ----- cross-wave softmax reduction scratch (D_ACC) -----
    // Every wave holds the whole Q_TILE_SIZE rows but only KV_TILE_SIZE of the
    // NUM_WARPS * KV_TILE_SIZE tokens one GEMM0 step consumes, so its row max and row sum
    // are partial and the waves have to merge them: one D_ACC per (query row, wave) for
    // each. Sits behind P in the same Q region, which is dead by the first softmax.
    static constexpr int smem_ml_elems           = W_M * T_N; // 128
    static constexpr size_t smem_ml_offset_bytes = smem_p_bytes;
    static constexpr size_t smem_ml_bytes        = 2 * smem_ml_elems * sizeof(D_ACC); // 1024
    static_assert(smem_ml_offset_bytes % (T_N * sizeof(D_ACC)) == 0,
                  "the T_N-wide row read wants a naturally aligned base");

    // ----- V transpose-read bank swizzle -----
    // A block holds smem_n_per_wave_kv tokens, one per row; the rows in its upper half carry
    // their d-groups XORed by one. Also the rotation modulus of the token deal, and hence a
    // shape in make_layout_kv_indices / make_layout_rkv -- keep it smem_n_per_wave_kv / 2.
    static constexpr int SWZ_D_BYTES = VEC_KV;                 // 16, one d-group
    static constexpr int SWZ_TOK_BIT = smem_n_per_wave_kv / 2; // 8, block row bit 3

    // Q, then the KV ring. P costs nothing on top: it aliases the Q region, which is dead once
    // the prologue has read Q into registers, and needs no ring of its own since a tile's
    // scores are written and consumed inside one phase.
    static constexpr size_t smem_bytes() { return 2 * smem_kv_bytes + smem_q_padding_bytes; }
    static_assert(smem_p_bytes + smem_ml_bytes <= smem_q_padding_bytes,
                  "P and the softmax scratch must fit in the Q region they alias");

    // fp8 nope + fp8 rope: one dwordx4 (16 fp8) per thread each -> 2 + 1 loads.
    static constexpr int q_nope_buffer_load_insts =
        (Q_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_Q); // 2
    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_HEAD_SIZE) / (WARP_SIZE * VEC_KV); // 9
    static constexpr int q_nope_ds_read_insts =
        (Q_TILE_SIZE * D_NOPE_SIZE) / (WARP_SIZE * VEC_Q);                                      // 8
    static constexpr int k_ds_read_insts = (KV_TILE_SIZE * D_HEAD_SIZE) / (WARP_SIZE * VEC_KV); // 9
    // One ds_read_b64_tr_b8 per (d-tile, VEC_TR_V tokens): NUM_D_SLICES x W_K / (W_N / VEC_TR_V).
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V); // 16
};

// ============================================================================
// Traits for the 16mx1 / 32nx4 variant: the M side of 16mx1 (one Q tile every wave shares,
// staged through LDS) on the N side of 16mx8 (a wave owns KV_TILE_SIZE = 32 tokens and reads
// them out of the 16mx8 K LDS image), with NUM_WARPS = 4 waves tiling N.
//
// A KV tile is KV_TILE_SIZE * NUM_WARPS = 128 tokens, cut into NUM_WARPS token GROUPS of 32.
// Group g is laid out in LDS exactly as 16mx8 lays out its whole 32-token tile -- same
// smem_n_per_wave / smem_n_rpt line blocks, same V-transpose bank swizzle -- so every
// GEMM0 / V read layout carries over unchanged with a group base added. Only the DMA differs:
// 16mx8 spreads one 32-token tile over 8 waves, here wave g fills group g on its own.
//
// What a wave then owns: GEMM0 over its own 32 tokens (2 e_n x 6 e_k = 12 MFMA, the 16mx8
// count) and GEMM1 over all 128 tokens x WAVE_D = 128 of the output d (4 k-steps x 4 d-slices
// x 2 = 32 MFMA, also the 16mx8 count). GEMM1 contracting tokens the wave did not score is
// what makes P round-trip LDS, as in 16mx1.
// ============================================================================
template <int Q_TILE_SIZE_  = 16,
          int KV_TILE_SIZE_ = 32,
          int NUM_WARPS_    = 4,
          typename D_Q_     = fp8_t,
          typename D_K_     = fp8_t,
          typename D_OUT_   = bf16_t,
          bool CAUSAL_      = false,
          bool LARGE_KV_    = false,
          int KV_SLOTS_     = 2>
struct opus_mla_decode_fp8_16mx1_32nx4_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_; // tokens in one K LDS group
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    static constexpr bool LARGE_KV    = LARGE_KV_;
    // One token group per wave: a wave scores its own KV_TILE_SIZE tokens and a tile is
    // NUM_WARPS groups.
    static constexpr int GROUPS_PER_TILE = NUM_WARPS_;
    // 2 = ping-pong: tile t+1 is DMA'd into the other slot while tile t is computed, which
    // is the whole point of the pipelined build. 1 = the single-slot floor, every tile's
    // gmem latency exposed. A slot is a whole 128-token tile at d = 576, so 2 is also the
    // ceiling: 2 * 74 KB + P leaves nothing, and the block drops to one per CU (4 waves,
    // one per SIMD) where the single-slot build keeps two blocks and 2 waves per SIMD.
    static constexpr int KV_SLOTS = KV_SLOTS_;
    static_assert(KV_SLOTS == 1 || KV_SLOTS == 2, "KV_SLOTS must be 1 or 2");
    // page_size 1 only: the rope DMA deals token lane_id / 4 of a 16-token line, so
    // token % PAGE_SIZE would be per-lane above 1 and could not ride a wave-uniform offset.
    static constexpr int PAGE_SIZE = 1;

    static constexpr int WARP_SIZE  = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE; // 256

    static constexpr int D_NOPE_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE; // 576, combined row stride

    using D_Q   = D_Q_;
    using D_K   = D_K_;
    using D_V   = D_K_;
    using D_OUT = D_OUT_;
    using D_ACC = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS; // the waves tile GEMM1's output d
    static constexpr int T_K = 1;

    static constexpr int W_M      = 16;
    static constexpr int W_N      = 16;
    static constexpr int W_K_NOPE = 128; // 16x16x128 f8f6f4 for QK nope
    static constexpr int W_K_ROPE = 32;  // plain fp8 16x16x32 for QK rope and PV

    static constexpr int KV_TILE_TOKENS = KV_TILE_SIZE * GROUPS_PER_TILE; // 128

    static constexpr int SLICE_D      = 32;                      // d per mma1 call
    static constexpr int WAVE_D       = D_NOPE_SIZE / NUM_WARPS; // 128, this wave's output d
    static constexpr int NUM_D_SLICES = WAVE_D / SLICE_D;        // 4

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;      // 1
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N;     // 2 e_n token tiles per wave
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_SIZE / W_K_NOPE; // 4
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE; // 2

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M; // 1
    static constexpr int GEMM1_E_N = SLICE_D / W_N;     // 2
    static constexpr int GEMM1_E_K = 1;                 // one k-step per mma1
    // One k-step is one token group (W_K_ROPE == KV_TILE_SIZE), so the steps are the groups.
    static constexpr int GEMM1_K_STEPS = KV_TILE_TOKENS / W_K_ROPE;    // 4
    static constexpr int PV_STEPS      = GEMM1_K_STEPS * NUM_D_SLICES; // 16 mma1 calls
    static_assert(GEMM1_K_STEPS == GROUPS_PER_TILE, "a PV k-step must be one token group");

    static constexpr int VEC_Q_NOPE     = 16; // dwordx4
    static constexpr int VEC_Q_ROPE     = 8;
    static constexpr int VEC_KV_NOPE    = 16;
    static constexpr int VEC_KV_ROPE    = 8;  // rope MFMA operand read (ds_read_b64)
    static constexpr int VEC_KV_ROPE_LD = 16; // rope DMA: 16 tokens/line, 4 lanes per token
    static constexpr int VEC_TR_V       = 8;
    static constexpr int VEC_O          = 4;
    // P write and read are both one 8B half-row: a lane's two e_n packs are contiguous in
    // the layout below, so the write is one ds_write_b64 and each k-step's read one
    // ds_read_b64.
    static constexpr int VEC_READ_P  = 8;
    static constexpr int VEC_WRITE_P = 4 * GEMM0_E_N;

    static constexpr int dwordx4_size     = 16;
    static constexpr int smem_padding_32B = 32 / sizeof(D_K);

    // ----- Q LDS geometry (fp8), the 16mx1 scheme -----
    // smem_n_rpt_q waves cover the Q_TILE_SIZE rows (smem_n_per_wave_q each) and each walks
    // smem_d_rpt_q chunks of d, so the LDS image is the same 8 blocks 16mx1 builds with 8
    // waves -- block index d_chunk * smem_n_rpt_q + warp_id -- and make_layout_rq_nope is
    // 16mx1's with waves_d renamed to smem_d_rpt_q.
    static constexpr int smem_linear_wave_q = WARP_SIZE * dwordx4_size / sizeof(D_Q); // 1024
    static constexpr int smem_n_per_wave_q  = 4;
    static constexpr int smem_n_rpt_q       = Q_TILE_SIZE / smem_n_per_wave_q;        // 4
    static constexpr int smem_d_per_wave_q  = smem_linear_wave_q / smem_n_per_wave_q; // 256
    static constexpr int smem_d_rpt_q       = D_NOPE_SIZE / smem_d_per_wave_q;        // 2
    static constexpr int smem_q_block       = smem_linear_wave_q + smem_padding_32B;  // 1056
    static constexpr size_t smem_q_bytes =
        smem_n_rpt_q * smem_d_rpt_q * smem_q_block * sizeof(D_Q); // 8448
    static_assert(smem_n_rpt_q == NUM_WARPS, "one Q row group per wave");

    // ----- K LDS geometry (fp8): 16mx8's image, once per 32-token group -----
    static constexpr int D_128B_NOPE_SIZE      = 128 / sizeof(D_K);                      // 128
    static constexpr int smem_linear_wave_nope = WARP_SIZE * dwordx4_size / sizeof(D_K); // 1024
    static constexpr int smem_n_per_wave       = 8;                              // tokens per line
    static constexpr int smem_n_rpt            = KV_TILE_SIZE / smem_n_per_wave; // 4
    static constexpr int smem_d_rpt_nope       = D_NOPE_SIZE / D_128B_NOPE_SIZE; // 4
    static constexpr int smem_nope_line        = smem_linear_wave_nope + smem_padding_32B; // 1056
    static constexpr size_t smem_kv_nope_group_bytes =
        smem_n_rpt * smem_d_rpt_nope * smem_nope_line * sizeof(D_K); // 16896

    // rope: one DMA line is a whole dwordx4 per lane, i.e. 16 tokens x D_ROPE_SIZE, which is
    // exactly one GEMM0 e_n tile. 16mx8 uses 4-token b32 lines because it spreads a 32-token
    // tile over 8 waves; here one wave owns the group, and the wider line cuts the rope DMA
    // from 8 instructions to 2. What pays for the b64 read is a source-side d-group swizzle
    // (make_layout_gkv_rope), the same trick the nope path uses for V.
    static constexpr int smem_linear_wave_rope = WARP_SIZE * VEC_KV_ROPE_LD / sizeof(D_K); // 1024
    static constexpr int rope_toks_per_line    = smem_linear_wave_rope / D_ROPE_SIZE;      // 16
    static constexpr int smem_n_rpt_rope       = KV_TILE_SIZE / rope_toks_per_line;        // 2
    // No padding: a rope read never spans two lines, one line being one e_n tile.
    static constexpr int smem_rope_line = smem_linear_wave_rope;
    static constexpr size_t smem_kv_rope_group_bytes =
        smem_n_rpt_rope * smem_rope_line * sizeof(D_K); // 2048
    static_assert(rope_toks_per_line == W_N, "a rope line must be exactly one GEMM0 e_n tile");

    static constexpr size_t smem_kv_nope_bytes =
        GROUPS_PER_TILE * smem_kv_nope_group_bytes; // 67584
    static constexpr size_t smem_kv_rope_bytes = GROUPS_PER_TILE * smem_kv_rope_group_bytes; // 8192
    // One slot: all NUM_WARPS token groups' nope regions, then their rope regions.
    static constexpr size_t smem_kv_slot_bytes = smem_kv_nope_bytes + smem_kv_rope_bytes; // 75776
    static constexpr size_t smem_kv_bytes      = KV_SLOTS * smem_kv_slot_bytes;

    // ----- V transpose-read bank swizzle (16mx8's, unchanged) -----
    static constexpr int SWZ_D_BYTES = VEC_KV_NOPE;         // 16, one d-group
    static constexpr int SWZ_TOK_BIT = smem_n_per_wave / 2; // 4, token-in-line bit 2

    // ----- P LDS geometry (fp8) -----
    // One block per wave, 16 query rows on a P_ROW_PITCH pitch. A row holds that wave's 32
    // scores in the order GEMM1 reads them back: position 8 * (lane / W_M) + 4 * e_n + pack
    // element, which is at once one lane's C register order (so the write is a single
    // ds_write_b64) and the MFMA A-operand k order (so the read is one ds_read_b64 per token
    // group). Both sides are then the same access -- 16 rows x four 8B halves -- and a pitch
    // of 6 slots plus the half XOR on the upper 8 rows tiles all 32 banks on both.
    // One block per token GROUP (a GEMM1 k-step reads exactly one), not per wave.
    static constexpr int P_ROW_PITCH     = 48;
    static constexpr int smem_p_pitch    = W_M * P_ROW_PITCH;              // 768 per block
    static constexpr size_t smem_p_bytes = GROUPS_PER_TILE * smem_p_pitch; // 3072
    static_assert(P_ROW_PITCH >= KV_TILE_SIZE && P_ROW_PITCH % 16 == 0 &&
                      (P_ROW_PITCH / 8) % 4 != 0,
                  "the row pitch must be an even, non-multiple-of-4 count of 8B slots");

    // ----- cross-wave softmax reduction scratch (D_ACC) -----
    static constexpr int smem_ml_elems    = W_M * T_N;                         // 64
    static constexpr size_t smem_ml_bytes = 2 * smem_ml_elems * sizeof(D_ACC); // 512

    // Q aliases the LAST KV slot: it is staged once per work item and is dead before that
    // slot is first written, with the barrier after the Q read covering the hazard. The last
    // slot rather than the first so that at KV_SLOTS = 2 the prologue can put tile_begin's
    // DMA in flight (slot 0) while Q is still landing. At KV_SLOTS = 1 this is what keeps
    // the block under 80 KB, i.e. two blocks per CU instead of one.
    static constexpr size_t smem_kv_offset = 0;
    static constexpr size_t smem_q_offset  = (KV_SLOTS - 1) * smem_kv_slot_bytes;
    static constexpr size_t smem_p_offset  = smem_kv_bytes;
    static constexpr size_t smem_ml_offset = smem_p_offset + smem_p_bytes;
    // 79360 at one slot, 155136 at two.
    static constexpr size_t smem_bytes() { return smem_ml_offset + smem_ml_bytes; }
    static_assert(smem_q_bytes <= smem_kv_slot_bytes, "Q must fit in the slot it aliases");
    static_assert(smem_kv_nope_group_bytes % 32 == 0 && smem_kv_rope_group_bytes % 32 == 0 &&
                      smem_nope_line % 32 == 0 && smem_kv_slot_bytes % 32 == 0,
                  "the V swizzle folds an XOR into a stride, so group bases must be 32B aligned");
    static_assert(smem_bytes() <= 160 * 1024, "gfx950 gives a workgroup 160 KB of LDS");

    // Per-thread instruction counts (waitcnt budgets).
    static constexpr int q_nope_buffer_load_insts =
        (Q_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_Q_NOPE); // 2
    // A wave fills its whole group: every nope and rope line of it.
    static constexpr int nope_lines_per_wave       = smem_n_rpt;                            // 4
    static constexpr int rope_lines_per_wave       = smem_n_rpt_rope;                       // 2
    static constexpr int kv_nope_buffer_load_insts = nope_lines_per_wave * smem_d_rpt_nope; // 16
    static constexpr int kv_rope_buffer_load_insts = rope_lines_per_wave;                   // 2
    static constexpr int kv_buffer_load_insts =
        kv_nope_buffer_load_insts + kv_rope_buffer_load_insts; // 18
    // One vector load for this wave's nope lines (their page indices are consecutive
    // kv_indices entries), one dword per rope line.
    static constexpr int kv_idx_load_insts = 1 + rope_lines_per_wave; // 3
    static constexpr int k_nope_ds_read_insts =
        (GEMM0_E_N * W_N * W_K_NOPE) / (WARP_SIZE * VEC_KV_NOPE); // 4
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * W_N * W_K_ROPE) / (WARP_SIZE * VEC_TR_V); // 2
};
