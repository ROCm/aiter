// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shared host/device ABI for the MXFP8 bpreshuffle pipelines and their COs.
struct opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 {
    const void *__restrict__ ptr_a;
    const void *__restrict__ ptr_b;
    void *__restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void *__restrict__ ptr_sfa;
    const void *__restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};
static_assert(sizeof(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) == 96);

// Four Wave64s use pinned A/B and selected accumulators with two LDS stages.
template<bool PadM = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_base_traits {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(NUM_WAVES == 4);
    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    // Bound matrix accesses and guard scale loads for the final partial M tile.
    static constexpr bool PAD_M = PadM;
    static constexpr bool GUARD_SFB_LANE = false;
    static constexpr bool RESET_SFA_BEFORE_LOAD = false;
    static constexpr int LOOP_UNROLL = 2;
    static constexpr int FIXED_K = 0;

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    static constexpr int VEC_SCALE_SF_PAIR = 8;

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    // Each input K128 scale is broadcast to the four hardware K32 lane groups.
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;
    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
    static_assert(NUM_KGROUPS == 4);

    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;

    // LDS capacity in K128 scale groups.
    static constexpr int SCALE_PANEL_K_CAPACITY = 64;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;

    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(VEC_SCALE_SF == E_M);
    static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};

template<class Base, int Panel>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits : Base {
    static_assert(Panel == 16 || Panel == 32 || Panel == 64);
    static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / Base::SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_BYTES = Base::SFA_PANEL_PITCH * Panel;
    static constexpr int SFB_PANEL_BYTES = Base::SCALE_N_HALVES * Panel * Base::VEC_SF;
    static constexpr int LDS_BYTES = Base::SMEM_A_ELEMS + Base::SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
};

template<int Panel, int FixedK, bool Reset, int Unroll>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_panel_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_scale_panel_traits<
        opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_base_traits<true>, Panel> {
    static constexpr int FIXED_K = FixedK;
    static constexpr int LOOP_UNROLL = Unroll;
    static constexpr bool GUARD_SFB_LANE = true;
    static constexpr bool RESET_SFA_BEFORE_LOAD = Reset;

    static_assert(Unroll == 2 || Unroll == 4);
    static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= 16384));
};

enum class opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view { tile, matrix, remaining_rows };
enum class opus_gemm_blockscale_mxscale_bpreshuffle_pin_output { lds, direct, stage_all };

template<class Base,
    opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view MatrixView =
        opus_gemm_blockscale_mxscale_bpreshuffle_pin_matrix_view::tile,
    opus_gemm_blockscale_mxscale_bpreshuffle_pin_output Output =
        opus_gemm_blockscale_mxscale_bpreshuffle_pin_output::lds,
    bool UnsignedThreadIndex = false, bool WideOutput = false, int GroupM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits : Base {
    static constexpr auto MATRIX_VIEW = MatrixView;
    static constexpr auto OUTPUT = Output;
    static constexpr bool UNSIGNED_THREAD_INDEX = UnsignedThreadIndex;
    static constexpr bool WIDE_OUTPUT = WideOutput;
    static constexpr int GROUPED_M = GroupM;
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8 || GroupM == 16);

    static constexpr int STATIC_M = 0;
    static constexpr int STATIC_N = 0;
    static constexpr int STATIC_K = 0;
    static constexpr bool STATIC_BATCH_STRIDES = false;
};

// Contiguous matrix and scale strides derived from M/N/K.
template<class Base, int M, int N, int K, bool FixedBatchStrides = true>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_fixed_traits : Base {
    static constexpr int STATIC_M = M;
    static constexpr int STATIC_N = N;
    static constexpr int STATIC_K = K;
    static constexpr bool STATIC_BATCH_STRIDES = FixedBatchStrides;
};
