// Kernel argument block (kernarg) for the fused FMHA forward shader.
//
// This is the core forward kernarg: tensors, strides and scale the per-block
// forward pass reads.  It is embedded as `base` in FmhaFwdSplitParams (the
// by-value argument of the split entry points) and read on the device by
// fmha_fwd_d64_device() in fused/pipeline.hpp.
//
// Layout note: the field order here IS the kernarg layout the HSACO expects,
// so do not reorder fields without re-checking the kernel ABI.

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

    // Softmax scale, PRE-MULTIPLIED by log2(e): scale == log2(e)/sqrt(head_dim),
    // NOT a plain 1/sqrt(head_dim).  The kernel's softmax is base-2 (it uses
    // exp2, not exp), so folding log2(e) into the scale converts the natural-e
    // softmax into the equivalent base-2 form.  See op_softmax.hpp.
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
// Split-K runs the forward attention G times over disjoint KV ranges, each pass
// writing a *normalized* fp32 partial output + a per-row natural-log LSE into a
// "scratch" staging buffer.  The combine pass then reweights those G partials
// back into the single global-softmax output (see op_combine.hpp for the
// math) and stores the final BF16 O.  This struct is the by-value argument the
// combine __global__ (fmha_fwd_d64_bf16_combine in the entry .cu files) reads.
//
// Scratch layout is "split-major": the G partial planes are the outermost axis,
// so plane g for the whole (B,Hq,Sq) problem is contiguous before plane g+1.
//   scratch_o  index of (g,b,h,row,d) =
//       (((g*B + b)*Hq + h)*Sq + row)*64 + d        (fp32, 64 = head_dim)
//   scratch_lse index of (g,b,h,row) =
//       ((g*B + b)*Hq + h)*Sq + row                 (fp32)
// (B and Hq are recovered on the device side from nhead_q + the grid; only the
// strides the kernel actually needs to write O are passed explicitly below.)
struct FmhaFwdCombineParams {
    const float* scratch_o;    // [G][B][Hq][Sq][64] fp32, split-major
    const float* scratch_lse;  // [G][B][Hq][Sq]      fp32
    __hip_bfloat16* o;         // final output, same layout as FmhaFwdParams.o
    float* lse;                // optional global LSE out (nullptr to skip)
    int num_splits;            // G
    int seqlen_q, nhead_q;
    int stride_o, nhead_stride_o, batch_stride_o;
    float scale;               // params.scale (base-2, log2e-folded) — for global LSE only
    // OPTIONAL fp32 output (split-K combine precision check). When non-null, the
    // combine ALSO writes the exact fp32 convex-combination result (before bf16
    // truncation) here, in NATURAL head-dim order, CONTIGUOUS [B][Hq][Sq][64]:
    //   o_fp32 index (b,h,R,d) = (((b*Hq + h)*Sq + R)*64 + d
    // This is the un-truncated O the bf16 store rounds — a caller can check it
    // at ~1e-5 to catch reweight-weight bugs the bf16 (~1e-3)
    // bound would hide. nullptr (the default for all value-init `cp{}` callers)
    // disables it → the bf16 path is byte-identical.
    float* o_fp32 = nullptr;
};

// Kernarg block for the split-K *forward* pass (the IsSplit=true variant of the
// fused forward kernel).
//
// Split-K runs the SAME per-block forward pass as a full (non-split) forward, but
// each block walks only a disjoint sub-range of the KV axis (its "split") and,
// instead of bf16-truncating O straight to the final tensor, writes a NORMALIZED
// fp32 partial output O_g + a per-row natural-log LSE_g into the split-major
// "scratch" staging buffer (same layout FmhaFwdCombineParams documents). The
// combine pass (op_combine.hpp) then folds the G partials into the final O.
//
// This struct is the by-value argument the split-forward __global__
// (fmha_fwd_d64_bf16_msk{0,1}_split in the entry .cu files) receives. It simply
// CARRIES the core forward kernarg (base) plus the split-only extras; the device
// function fmha_fwd_d64_device() takes a `const FmhaFwdParams&` (== base) plus
// the split inputs as trailing arguments. See pipeline.hpp.
//
// Scratch layout is the SAME split-major layout FmhaFwdCombineParams documents:
//   scratch_o  (split_idx,b,h,row,d) =
//       (((split_idx*B + b)*Hq + h)*Sq + row)*64 + d   (fp32, 64 = head_dim)
//   scratch_lse(split_idx,b,h,row)   =
//        ((split_idx*B + b)*Hq + h)*Sq + row           (fp32)
// (B and Hq are recovered device-side: Hq == base.nhead_q, and the split grid's
// z-axis is batch*num_splits so B == gridDim.z / num_splits. See pipeline.hpp's
// epilogue base-pointer computation.)
struct FmhaFwdSplitParams {
    FmhaFwdParams base;        // the ordinary forward kernarg (tensors, strides, scale)
    float* scratch_o;          // [G][B][Hq][Sq][64] fp32, split-major (partial O_g)
    float* scratch_lse;        // [G][B][Hq][Sq]      fp32 (natural-log LSE_g)
    int num_splits;            // G (KV axis is partitioned into G disjoint ranges)
    // split_idx: which of the G splits this launch handles. The shipping globals
    // DECODE the split index from blockIdx.z (grid z-axis is batch*num_splits;
    // split_idx = blockIdx.z % num_splits — see the entry .cu files), so this field is
    // redundant in the current dispatch; it is kept for completeness so a host
    // caller could instead pass the split index explicitly. The device function
    // takes split_idx as an argument either way.
    int split_idx;
};

// --- Compile-time tile / launch geometry (D64 BF16 kernel specific) ---
// These describe how the fused kernel partitions the problem and lays out LDS.
// The host launcher also reads kM0 (M-tile size) and kBlockSize to build the grid.
// Tile constants (D64 bf16 specific)
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
