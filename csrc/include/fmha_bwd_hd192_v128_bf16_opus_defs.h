// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel arguments of the OPUS backward of the D_QK = 192 / D_V = 128 attention, whose output
// may be gated: out = attention(q, k, v) * sigmoid(gate).
#pragma once

// Pre-pass, one row per (token, head) of D_V = 128:
//   s = sigmoid(gate), dO = dout * s, dgate = dout * out * (1 - s), D = rowsum(dout * out)
// D equals the usual rowsum(dO * O) of the ungated attention, since out = O * s.
// The gate is a compile-time option of the kernel: opus_fmha_bwd_v128_pre_gate_kernel does the
// above, opus_fmha_bwd_v128_pre_kernel makes D alone and leaves ptr_gate, ptr_do and ptr_dgate
// unused.
// Strides are in elements; D is [H, total] fp32 like the forward's LSE.
struct fmha_bwd_v128_pre_kargs
{
    const void* __restrict__ ptr_dout;
    const void* __restrict__ ptr_out;
    const void* __restrict__ ptr_gate;
    void* __restrict__ ptr_do;
    void* __restrict__ ptr_dgate;
    float* __restrict__ ptr_d;
    int total;
    int H;
    int stride_dout_n, stride_dout_h;
    int stride_out_n, stride_out_h;
    int stride_gate_n, stride_gate_h;
    int stride_do_n, stride_do_h;
    int stride_dgate_n, stride_dgate_h;
    int stride_d_h;
};

static constexpr int FMHA_BWD_V128_PRE_BLOCK = 256;

namespace fmha_bwd_d192_v128 {
// Heads per scheduling group of the attention's backward: a launch's workgroups go group by group
// (fmha_bwd_d192_v128_kargs). The GPU deals workgroups by id to 32 lanes, a round of 32 ids at a
// time. With a causal mask a sequence's blocks differ in length and a group is 32 heads, so that
// a round is 32 heads' same block; without a mask a group is 16 heads and a round holds two
// consecutive blocks.
constexpr int HG        = 16; // heads per scheduling group
constexpr int HG_CAUSAL = 32; // with a causal mask
__host__ __device__ inline int wg_heads(int causal) { return causal ? HG_CAUSAL : HG; }
} // namespace fmha_bwd_d192_v128

// The attention's backward (opus_fmha_bwd_d192_v128_kernel), group / varlen: the dK / dV pass and
// the dQ pass in one launch, each on 4 waves, one per SIMD. A dK / dV workgroup is a block of 192
// keys of a (sequence, head): per tile of 32 queries it recomputes S^T = K Q^T and dP^T = V dO^T,
// forms P and dS = P (dP - D) from the forward's LSE and the pre-pass's D, and accumulates dK and
// dV in registers. A dQ workgroup is a block of 256 queries, or of 192: per tile of 32 keys it
// recomputes S and dP likewise and accumulates dQ = dS K.
// q [total_q, H, 192] and d_o [total_q, H, 128] are bf16, a row per head; k [total_k, H_k, 192]
// and v [total_k, H_k, 128] likewise, H a multiple of H_k: head h reads the K and V of head
// h / (H / H_k). The outputs dq [total_q, H, 192], dk [total_k, H, 192] and dv [total_k, H, 128]
// are bf16, a row per head h. dK and dQ include softmax_scale. Strides are in elements. The
// arguments are 192 bytes, which the kernel reads in scalar loads. A pass with no blocks is left
// out.
//
// The launch's grid is the workgroup's place, so the kernel divides nothing: x is the head within
// its group of fmha_bwd_d192_v128::wg_heads heads, and a pass's blocks go in order, the long ones
// first. The order of the passes goes by the mask:
//   causal   z = group * num_seqs + sequence, y the block (the dK / dV pass's, then the dQ
//            pass's): a group's two passes on a sequence run back to back.
//   none     z = sequence, y the dK / dV pass's rows, then the dQ pass's; a pass's row is
//            group * nb + block (nb the pass's block count): a sequence's dK / dV pass group by
//            group, then its dQ pass.
// A last group of r heads, fewer than a whole group, takes its ids in order instead: read row by
// row (a row is a y of a pass), they are its (block, head) pairs, r heads to a block, so that every
// id of a row has a head, but for the last row of a pass.
// Ids past a pass's blocks end at once.
struct fmha_bwd_d192_v128_kargs
{
    const void* __restrict__ ptr_q;
    const void* __restrict__ ptr_k;
    const void* __restrict__ ptr_v;
    const void* __restrict__ ptr_do;
    const float* __restrict__ ptr_lse;
    const float* __restrict__ ptr_d;
    const int* __restrict__ ptr_seqstart_q;
    const int* __restrict__ ptr_seqstart_k;
    void* __restrict__ ptr_dq;
    void* __restrict__ ptr_dk;
    void* __restrict__ ptr_dv;
    int H;
    int num_seqs;
    int n_kv_blocks; // dK / dV pass: key blocks per sequence, from max_seqlen_k
    int n_q_blocks;  // dQ pass: query blocks per sequence, from max_seqlen_q
    int rem_magic;   // ceil(2^31 / r), r the heads of the last group: c * rem_magic >> 31 is c / r
    int causal;
    float softmax_scale;
    int stride_q_n, stride_q_h;
    int stride_k_n, stride_k_h;
    int stride_v_n, stride_v_h;
    int stride_do_n, stride_do_h;
    int stride_lse_h; // LSE and D share [H, total_q]
    int stride_dq_n, stride_dq_h;
    int stride_dk_n, stride_dk_h;
    int stride_dv_n, stride_dv_h;
    int div_magic;   // ceil(2^31 / d): c * div_magic >> 31 is c / d. d = num_seqs (causal) or
                     // n_kv_blocks
    int div_magic_q; // the same for the dQ pass: d = num_seqs (causal) or n_q_blocks
    int kv_rows;     // without a mask: the rows (y) of the dK / dV pass
    int kv_magic;    // ceil(2^31 / s), s = H / H_k: h * kv_magic >> 31 is h / s
};
static_assert(sizeof(fmha_bwd_d192_v128_kargs) == 192, "the kernel's argument loads");

// The grid of that launch (0 in x: nothing to launch), the magics and kv_rows, from the other
// fields and kv_share = H / H_k; false if a division of the kernel's would not be exact (the
// dividend times the divisor must stay below 2^31).
__host__ inline bool
fmha_bwd_d192_v128_grid(fmha_bwd_d192_v128_kargs& a, int kv_share, unsigned int grid[3])
{
    const int hg           = fmha_bwd_d192_v128::wg_heads(a.causal);
    const long long groups = (a.H + hg - 1) / hg;
    const long long rest   = a.H - (groups - 1) * hg; // heads of the last group
    const long long xw     = a.H < hg ? a.H : hg;     // ids of a row
    const long long nb     = a.n_kv_blocks > a.n_q_blocks ? a.n_kv_blocks : a.n_q_blocks;
    const int passes       = (a.n_kv_blocks > 0) + (a.n_q_blocks > 0);
    const int by_group = a.causal ? 1 : 0; // the order of the passes, see fmha_bwd_d192_v128_kargs
    const long long d  = by_group ? a.num_seqs : nb;
    // Without a mask: the rows of a pass of n blocks.
    auto rows  = [&](long long n) { return (groups - 1) * n + (n * rest + xw - 1) / xw; };
    auto magic = [](long long v) {
        return v > 0 ? static_cast<int>(((1LL << 31) + v - 1) / v) : 0;
    };
    grid[0]   = static_cast<unsigned int>(xw);
    grid[1]   = static_cast<unsigned int>(by_group ? a.n_kv_blocks + a.n_q_blocks
                                                 : rows(a.n_kv_blocks) + rows(a.n_q_blocks));
    grid[2]   = static_cast<unsigned int>(by_group ? groups * a.num_seqs : a.num_seqs);
    a.kv_rows = by_group ? 0 : static_cast<int>(rows(a.n_kv_blocks));
    if(passes == 0 || a.num_seqs <= 0)
        grid[0] = 0;
    a.div_magic   = magic(by_group ? a.num_seqs : a.n_kv_blocks);
    a.div_magic_q = magic(by_group ? a.num_seqs : a.n_q_blocks);
    a.rem_magic   = magic(rest);
    a.kv_magic    = magic(kv_share);
    return groups * d * d < (1LL << 31) && nb * xw * rest < (1LL << 31) &&
           static_cast<long long>(a.H) * kv_share < (1LL << 31);
}

// The dK / dV pass (opus_fmha_bwd_d192_v128_kernel): each wave (one per SIMD) owns 48 keys with
// their K, V, dK and dV (accumulators in AGPRs), on 32-query tiles so that every GEMM is a
// full-rate 16x16x32 MFMA. n_kv_blocks counts blocks of BK = 192 keys.
namespace fmha_bwd_d192_v128_dkdv {
constexpr int BK   = 192; // keys per workgroup
constexpr int NW   = 4;   // waves; wave w owns keys [48 w, 48 w + 48)
constexpr int WR   = BK / NW;
constexpr int BQ   = 32; // queries per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
// LDS, in bf16 elements: NBUF query tiles, each sQ [BQ][D_QK] (B of S^T; B of dK, transposed),
// sDO [BQ][D_V] (B of dP^T; B of dV, transposed) and the tile's LSE and D in fp32 (256 bytes
// each, one DMA load). Rows padded by 16 elements: with 8, both the B-fragment ds_read_b128 and
// the ds_read_b64_tr_b16 reads conflict 2-way. The epilogue then transposes each wave's
// [48][dK], then its [48][dV], through the same LDS.
constexpr int NBUF = 4;
constexpr int LD_Q = D_QK + 16, LD_DO = D_V + 16, LD_O = D_QK + 16;
constexpr int OFF_Q = 0, OFF_DO = OFF_Q + BQ * LD_Q, OFF_LSE = OFF_DO + BQ * LD_DO,
              OFF_D     = OFF_LSE + 128;
constexpr int SZ_BUF    = OFF_D + 128;
constexpr int SMEM_MAIN = NBUF * SZ_BUF, SMEM_EPI = NW * WR * LD_O;
constexpr int SMEM_ELEMS = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
} // namespace fmha_bwd_d192_v128_dkdv

// The dQ pass (opus_fmha_bwd_d192_v128_kernel): wave w (one per SIMD) owns queries
// [64 w, 64 w + 64) of the workgroup's 256: its dO and Q's last 64 dims in VGPRs, Q's first 128
// dims and its dQ in AGPRs. A sequence's last blocks are of 192 queries (48 a wave) where that
// covers it with fewer rows to spare. n_q_blocks counts blocks of BQ = 256 queries: the blocks of
// the longest sequence.
namespace fmha_bwd_d192_v128_dq {
constexpr int NW   = 4;       // waves
constexpr int WQ   = 64;      // queries per wave
constexpr int BQ   = NW * WQ; // queries per workgroup
constexpr int KC   = 32;      // keys per tile
constexpr int D_QK = 192, D_V = 128, D_NOPE = 128, D_ROT = 64;
constexpr int BLOCK = NW * 64;
// LDS, in bf16 elements: NBUF key tiles, each sKN [KC][D_NOPE] and sKR [KC][D_ROT] (K; B of S, and
// of dQ transposed) and sV [KC][D_V] (B of dP): K's first 128 dims and its last 64 are a DMA
// region each. Rows padded by 16 elements (conflict-free, see fmha_bwd_d192_v128_dkdv). The
// epilogue then transposes each wave's [64][dQ] through the same LDS.
constexpr int NBUF  = 4;
constexpr int LD_KN = D_NOPE + 16, LD_KR = D_ROT + 16, LD_V = D_V + 16, LD_O = D_QK + 16;
constexpr int OFF_KN = 0, OFF_KR = OFF_KN + KC * LD_KN, OFF_V = OFF_KR + KC * LD_KR;
constexpr int SZ_BUF    = OFF_V + KC * LD_V;
constexpr int SMEM_MAIN = NBUF * SZ_BUF, SMEM_EPI = NW * WQ * LD_O;
constexpr int SMEM_ELEMS = SMEM_MAIN > SMEM_EPI ? SMEM_MAIN : SMEM_EPI;
} // namespace fmha_bwd_d192_v128_dq
