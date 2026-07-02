// Kernel argument block (kernarg) for the fused FMHA forward shader: tensors,
// strides and scale the per-block forward pass reads. Embedded as `base` in
// FmhaFwdSplitParams and read on the device by fmha_fwd_d64_device()
// (fused/pipeline.hpp).
//
// Layout note: the field order IS the kernarg layout the HSACO expects — do not
// reorder without re-checking the kernel ABI.

#pragma once
#include <hip/hip_bf16.h>
#include <cstdint>

struct FmhaFwdParams {
    // Input tensors, row-major, BF16.  Logical layout per tensor is
    // [batch, nhead, seqlen, head_dim]; the actual element offset of any
    // (b, h, s, d) is b*batch_stride + h*nhead_stride + s*stride + d.
    const __hip_bfloat16 *q, *k, *v;
    // Output tensor O, same [batch, nhead_q, seqlen_q, head_dim] layout as Q.
    __hip_bfloat16* o;
    // Optional log-sum-exp output, one FP32 value per query row
    // ([batch, nhead_q, seqlen_q]).  nullptr disables LSE writes.
    float* lse;

    // Per-sequence lengths.  In batch mode these apply to every sequence; in
    // group/varlen mode they are upper bounds and the real per-batch lengths
    // come from seqstart_* (see below).
    int seqlen_q, seqlen_k;
    // Head counts.  nhead_q is the query head count; nhead_k is the KV head
    // count.  When nhead_q > nhead_k the kernel runs grouped-query attention
    // (each KV head is shared by nhead_q / nhead_k query heads).
    int nhead_q, nhead_k;

    // Softmax scale PRE-MULTIPLIED by log2(e): scale == log2(e)/sqrt(head_dim), NOT
    // plain 1/sqrt(head_dim). The kernel's softmax is base-2 (exp2), so folding
    // log2(e) in converts natural-e softmax to the base-2 form. See op_softmax.hpp.
    float scale;

    // All strides below are in ELEMENTS (BF16 units), not bytes.  For each
    // tensor: stride_* is the per-token (seqlen) stride, nhead_stride_* is the
    // per-head stride, batch_stride_* is the per-batch stride.  Contiguous
    // packing makes stride == head_dim, nhead_stride == seqlen*head_dim, etc.
    int stride_q, nhead_stride_q, batch_stride_q;
    int stride_k, nhead_stride_k, batch_stride_k;
    int stride_v, nhead_stride_v, batch_stride_v;
    int stride_o, nhead_stride_o, batch_stride_o;

    // Group (variable-length) mode: cumulative token-offset tables of length
    // batch+1, so the b-th sequence spans tokens [seqstart[b], seqstart[b+1]).
    // When non-null the kernel ignores batch_stride_* (sequences are packed
    // back-to-back) and derives each per-batch length from the table.  Both
    // nullptr selects fixed-length batch mode.
    //
    // Note: the causal "mask_shift" (seqlen_k - seqlen_q) is NOT stored here;
    // the kernel computes it on the fly in pipeline.hpp.
    const int32_t* seqstart_q;
    const int32_t* seqstart_k;
};

// Kernarg block for the split-K *combine* pass (fmha_fwd_d64_bf16_combine).
//
// Split-K runs the forward G times over disjoint KV ranges, each writing a
// normalized fp32 partial + per-row natural-log LSE into "scratch". Combine
// reweights the G partials back into the global-softmax output (math in
// op_combine.hpp) and stores final BF16 O.
//
// Scratch layout is "split-major": the G partial planes are the outermost axis
// (plane g contiguous before plane g+1).
//   scratch_o  index of (g,b,h,row,d) =
//       (((g*B + b)*Hq + h)*Sq + row)*64 + d        (fp32, 64 = head_dim)
//   scratch_lse index of (g,b,h,row) =
//       ((g*B + b)*Hq + h)*Sq + row                 (fp32)
// (B and Hq are recovered device-side from nhead_q + the grid; only the strides
// needed to write O are passed explicitly below.)
struct FmhaFwdCombineParams {
    const float* scratch_o;    // [G][B][Hq][Sq][64] fp32, split-major
    const float* scratch_lse;  // [G][B][Hq][Sq]      fp32
    __hip_bfloat16* o;         // final output, same layout as FmhaFwdParams.o
    float* lse;                // optional global LSE out (nullptr to skip)
    int num_splits;            // G
    int seqlen_q, nhead_q;
    int stride_o, nhead_stride_o, batch_stride_o;
    float scale;               // params.scale (base-2, log2e-folded) — for global LSE only
    // OPTIONAL fp32 output (split-K combine precision check). When non-null, combine
    // ALSO writes the exact fp32 convex-combination result (before bf16 truncation)
    // in NATURAL head-dim order, CONTIGUOUS [B][Hq][Sq][64]:
    //   o_fp32 index (b,h,R,d) = (((b*Hq + h)*Sq + R)*64 + d
    // The un-truncated O the bf16 store rounds — check at ~1e-5. nullptr (default for
    // value-init `cp{}` callers) disables it -> bf16 path byte-identical.
    float* o_fp32 = nullptr;
};

// Kernarg block for the split-K *forward* pass (IsSplit=true variant).
//
// Runs the SAME per-block forward as a full forward, but each block walks only a
// disjoint KV sub-range (its "split") and writes a normalized fp32 partial O_g +
// per-row natural-log LSE_g into split-major scratch (same layout
// FmhaFwdCombineParams documents); combine (op_combine.hpp) folds the G partials
// into final O. This struct CARRIES the core kernarg (base) plus split-only extras;
// fmha_fwd_d64_device() takes `const FmhaFwdParams&` (== base) plus the split inputs
// as trailing args.
//
// Scratch layout, same split-major as FmhaFwdCombineParams:
//   scratch_o  (split_idx,b,h,row,d) =
//       (((split_idx*B + b)*Hq + h)*Sq + row)*64 + d   (fp32, 64 = head_dim)
//   scratch_lse(split_idx,b,h,row)   =
//        ((split_idx*B + b)*Hq + h)*Sq + row           (fp32)
// (Device-side: Hq == base.nhead_q, split grid z-axis is batch*num_splits so
// B == gridDim.z / num_splits.)
struct FmhaFwdSplitParams {
    FmhaFwdParams base;        // the ordinary forward kernarg (tensors, strides, scale)
    float* scratch_o;          // [G][B][Hq][Sq][64] fp32, split-major (partial O_g)
    float* scratch_lse;        // [G][B][Hq][Sq]      fp32 (natural-log LSE_g)
    int num_splits;            // G (KV axis is partitioned into G disjoint ranges)
    // split_idx: which of the G splits this launch handles. The shipping globals
    // decode it from blockIdx.z (split_idx = blockIdx.z % num_splits), so this field
    // is redundant in the current dispatch; kept so a host caller could pass it
    // explicitly. The device function takes split_idx as an argument either way.
    int split_idx;
};

// --- Compile-time tile / launch geometry (D64 BF16 kernel specific) ---
// How the fused kernel partitions the problem and lays out LDS. The host launcher
// reads kM0 and kBlockSize to build the grid.
constexpr int kM0 = 128;          // query rows per M-tile (one threadblock's work in Q)
constexpr int kN0 = 64;           // key columns per K-tile (GEMM0 inner N)
constexpr int kK0 = 32;           // contraction depth per step of GEMM0 (Q.K^T)
constexpr int kN1 = 64;           // output columns per tile of GEMM1 (= head_dim)
constexpr int kK1 = 32;           // contraction depth per step of GEMM1 (P.V)
constexpr int kBlockSize = 256;   // threads per block (= kNumWarps * kWarpSize)
constexpr int kNumWarps = 4;      // warps (waves) per block
constexpr int kWarpSize = 64;     // lanes per wavefront (CDNA: 64, not 32)
constexpr int kHeadDim = 64;      // head dimension D this kernel is specialized for
constexpr int kKPack = 8;         // BF16 elements packed per vectorized LDS access
// LDS rows are padded to kPixelsPerRow + kKPack to avoid bank conflicts.
constexpr int kPixelsPerRow = 64;
constexpr int kPaddedRowStride = kPixelsPerRow + kKPack; // 72 elements
constexpr int kSingleSmemElements = 2304; // per LDS buffer, in bf16 elements
constexpr int kNumLdsBuffers = 3;         // triple-buffered Q/K/V staging in LDS
// Total LDS footprint in bytes (3 * 2304 * 2 = 13824).
constexpr int kLdsBytes = kNumLdsBuffers * kSingleSmemElements * sizeof(__hip_bfloat16); // 13824
