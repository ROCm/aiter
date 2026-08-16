#pragma once

#include <algorithm>

using bf16_t = __bf16;
using fp16_t = __fp16;
using fp8_t  = _BitInt(8);
using bf8_t  = unsigned _BitInt(8);

static constexpr int NUM_CU = 256;

struct mla_kargs
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

    const int* __restrict__ work_indptr;
    const int* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_page;
    float softmax_scale;
};

// ============================================================================
// Traits for the *combined* fp8 MLA decode kernel (mla_decode_fwd_16mx8_32nx1
// _fp8fp8_ps_opus.hpp).
//
// Differences vs. dsa_v32 (dsa_v32_16mx8_32nx1_fp8_traits in defs.h):
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
struct mla_16mx8_32nx1_fp8fp8_ps_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    // KV cache past the 4 GiB a buffer descriptor can address; see the KV load in
    // mla_decode_fwd_16mx8_32nx1_fp8fp8_ps_opus.hpp. Costs ~1-2% and 3 spilled VGPR, so
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

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
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

    // ----- K nope LDS geometry (fp8), identical scheme to dsa_v32 nope -----
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
    static constexpr int SWZ_D_BYTES = VEC_KV_NOPE;      // 16, one d-group
    static constexpr int SWZ_TOK_BIT = smem_n_per_wave / 2; // token-in-line bit 2

    // ----- K rope LDS geometry (fp8). rope d = 64 fits in a single sub-line
    //       so we treat one 64-wide "128B-like" chunk per row. -----
    static constexpr int D_128B_ROPE_SIZE      = D_ROPE_SIZE; // 64
    static constexpr int smem_linear_wave_rope = WARP_SIZE * VEC_KV_ROPE_LD / sizeof(D_K);
    static constexpr int smem_d_rpt_rope   = 1;
    static constexpr int smem_padding_rope = 16 / sizeof(D_K); // 16
    // New rope layout: one warp per LDS line (4 tokens/line), NUM_WARPS lines
    // total. The read (make_layout_rk_rope) reaches lines 4..7 via the GEMM0_E_N
    // stride, and the store (make_layout_sk_rope) writes lines 0..7 via warp_id,
    // so the region must hold NUM_WARPS lines, not smem_n_rpt.
    static constexpr size_t smem_k_rope_bytes =
        NUM_WARPS * smem_d_rpt_rope * (smem_linear_wave_rope + smem_padding_rope) * sizeof(D_K); //2176

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

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
          bool LARGE_KV_    = false>
struct mla_16mx1_16nx8_fp8fp8_ps_traits
{
    static constexpr int Q_TILE_SIZE  = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS    = NUM_WARPS_;
    static constexpr bool CAUSAL      = CAUSAL_;
    // KV cache past the 4 GiB a buffer descriptor can address; see the KV load in
    // mla_decode_fwd_16mx1_16nx8_fp8fp8_ps_opus.hpp. Costs ~1-2% and 3 spilled VGPR, so
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

    static constexpr int GEMM0_E_M      = Q_TILE_SIZE / W_M;  // 1
    static constexpr int GEMM0_E_N      = KV_TILE_SIZE / W_N; // 1
    static constexpr int GEMM0_E_K = D_HEAD_SIZE_PADDING / W_K;  // 5

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;       // 1
    static constexpr int GEMM1_E_N = NUM_D_SLICES;           // 4
    static constexpr int GEMM1_E_K = KV_TILE_SIZE * NUM_WARPS / W_K; // 1

    static constexpr int VEC_Q  = 16; // fp8 dwordx4 global/ds vector
    static constexpr int VEC_KV = 16;
    static constexpr int VEC_READ_P = 8; // one 8B half of a block row, i.e. a ds_read_b64
    static constexpr int VEC_WRITE_P = 4;
    static constexpr int VEC_TR_V       = 8;
    static constexpr int VEC_O          = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int dword_size = 4;

    // ----- Q LDS geometry (fp8) -----
    // One block of smem_linear_wave_q + smem_padding_32B per wave, holding that wave's
    // buffer_load_lds verbatim: smem_n_rpt_q waves cover the Q_TILE_SIZE rows
    // (smem_n_per_wave_q rows each) and the remaining smem_d_rpt_q cover d
    // (smem_d_per_wave_q each). See make_layout_gq in the kernel.
    static constexpr int smem_linear_wave_q = WARP_SIZE * dwordx4_size / sizeof(D_Q); // 1024
    static constexpr int smem_n_per_wave_q  = 4;
    static constexpr int smem_n_rpt_q       = Q_TILE_SIZE / smem_n_per_wave_q;          // 4
    static constexpr int smem_d_per_wave_q  = smem_linear_wave_q / smem_n_per_wave_q;    // 256
    static constexpr int smem_d_rpt_q       = D_NOPE_SIZE / smem_d_per_wave_q;           // 2
    static constexpr int smem_padding_32B   = 32 / sizeof(D_Q);
    static constexpr size_t smem_q_bytes    = smem_n_rpt_q * smem_d_rpt_q *
                                           (smem_linear_wave_q + smem_padding_32B) *
                                           sizeof(D_Q); // 8448B
    static_assert(smem_n_rpt_q * smem_d_rpt_q == NUM_WARPS, "one Q LDS block per wave");

    // ----- K LDS geometry (fp8) -----
    static constexpr int smem_linear_wave_kv = WARP_SIZE * dwordx4_size / sizeof(D_K); // 1024
    static constexpr int smem_n_per_wave_kv  = 16;
    static constexpr int smem_n_rpt_kv = KV_TILE_SIZE * NUM_WARPS / smem_n_per_wave_kv;   // 8
    static constexpr int smem_d_per_wave_kv = smem_linear_wave_kv / smem_n_per_wave_kv;   // 64
    static constexpr int smem_d_rpt_kv      = D_HEAD_SIZE / smem_d_per_wave_kv;           // 9
    static constexpr size_t smem_kv_bytes = smem_n_rpt_kv * smem_d_rpt_kv *
                                                 (smem_linear_wave_kv + smem_padding_32B) *
                                                 sizeof(D_K);

    // ----- P LDS geometry (fp8) -----
    // A block carries smem_linear_wave_p bytes of scores but sits on a wider pitch, and its rows
    // are permuted and half-swapped: together those three take both the write and the read to
    // zero bank conflicts, see make_layout_sp. The read forces the pitch to 128 mod 256, and the
    // cheapest such pitch leaves a third of a block spare.
    static constexpr int smem_linear_wave_p = WARP_SIZE * dword_size / sizeof(D_V); // 256
    static constexpr int P_SWZ_ROW_MUL      = 11; // rows land on 16 * (11 * r % W_M)
    static constexpr int smem_p_pitch       = 384;
    static constexpr size_t smem_p_bytes = NUM_WARPS * smem_p_pitch * sizeof(D_V); // 3072B
    static_assert(smem_p_pitch % 256 == 128 && smem_p_pitch >= smem_linear_wave_p,
                  "the two blocks a read phase spans must sit 16 bank slots apart");

    static constexpr size_t smem_q_padding_bytes = 9 * 1024 * sizeof(D_Q);
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
    static_assert(smem_p_bytes <= smem_q_padding_bytes, "P must fit in the Q region it aliases");

    // fp8 nope + fp8 rope: one dwordx4 (16 fp8) per thread each -> 2 + 1 loads.
    static constexpr int q_nope_buffer_load_insts =
        (Q_TILE_SIZE * D_NOPE_SIZE) / (BLOCK_SIZE * VEC_Q); // 2
    static constexpr int kv_buffer_load_insts =
        (KV_TILE_SIZE * D_HEAD_SIZE) / (WARP_SIZE * VEC_KV); // 9
    static constexpr int q_nope_ds_read_insts =
        (Q_TILE_SIZE * D_NOPE_SIZE) / (WARP_SIZE * VEC_Q); // 8
    static constexpr int k_ds_read_insts =
        (KV_TILE_SIZE * D_HEAD_SIZE) / (WARP_SIZE * VEC_KV); // 9
    // One ds_read_b64_tr_b8 per (d-tile, VEC_TR_V tokens): NUM_D_SLICES x W_K / (W_N / VEC_TR_V).
    static constexpr int v_ds_read_insts =
        (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V); // 16
};
