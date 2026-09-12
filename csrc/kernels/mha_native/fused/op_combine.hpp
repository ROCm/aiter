#pragma once
#include <hip/hip_bf16.h>
#include <math.h>
#include "runner/params.hpp"

// ================================================================
// op_combine.hpp — split-K COMBINE pass of the D64 FMHA fwd kernel
// ================================================================
//
// ROLE IN THE PIPELINE
//   Split-K runs the forward attention G times over G disjoint KV ranges. Each
//   pass ("split") writes a *normalized* fp32 partial output O_g and a per-row
//   natural-log LSE_g into a "scratch" staging buffer (split-major layout, see
//   FmhaFwdCombineParams in runner/params.hpp). The COMBINE pass — implemented
//   here — is a cheap second kernel that reads those G partials per row and folds
//   them back into the single global-softmax output, then truncates fp32 -> bf16
//   and stores the final O.
//
// THE MATH (natural-e domain)
//   For one output row, given G partials O_g[0..63] and scalars LSE_g:
//       M       = max_g LSE_g                       (global row max)
//       w_g     = (LSE_g == -inf) ? 0 : exp(LSE_g - M)   (max-subtract: stable)
//       denom   = Σ_g w_g                           (= exp(L* - M))
//       O[d]    = Σ_g (w_g / denom) * O_g[d]        (convex combination, Σ=1)
//   If EVERY range is -inf (M stays -inf) there is no mass -> O[d] = 0 (no NaN).
//   The fp32 reweighting is exact; the only lossy step is the final bf16 store.
//
// WHY NO swz HERE (★ critical layout fact)
//   The scratch is stored in NATURAL head-dim order: scratch_o[g][b][h][row][d]
//   has d == the natural head-dim index 0..63. (The split *producer*
//   epilog_store_split (op_epilog.hpp) un-swizzles the GEMM-inherited register
//   layout BEFORE writing scratch, so the combine consumes already-natural
//   planes.) Therefore the combine reads scratch plane element d and writes O
//   column d DIRECTLY: it must NOT run d through swz(). Applying swz here would
//   permute the columns and break the natural-order layout invariant.
//
// LAUNCH CONTRACT (mirrors the forward kernel)
//   Grid : dim3(nhead_q, m_tiles, batch)  — one block per (b, h, m_tile),
//          m_tiles = ceil(seqlen_q / kM0), kM0 = 128 query rows per tile.
//   Block: kBlockSize (=256) threads.
//   Decode: h = blockIdx.x, m_tile = blockIdx.y, b = blockIdx.z.
//   For an in-tile row index `row`, the absolute output row is
//          R = m_tile * kM0 + row,  valid only while R < seqlen_q.
//
// ROW -> THREAD MAPPING (deliberately the SIMPLEST correct one)
//   This pass is memory-light and runs once; it is NOT perf-gated (the combine
//   measures ~3% of GPU time). So we use the most obviously-correct mapping:
//       row = threadIdx.x         (one query row per thread)
//   The block has kBlockSize(=256) threads but a tile is only kM0(=128) rows, so
//   threads 0..127 each own exactly one row and threads 128..255 idle. Each
//   active thread loops g = 0..G-1 and d = 0..63, accumulates the convex
//   combination in fp32 registers, then bf16-truncates and stores its 64 values.
//   No cross-thread communication, no LDS — trivially race-free and correct.
//
// B>1 SUPPORT
//   The combine struct does not carry the batch count B, but the split-major
//   g-stride needs it. B == gridDim.z (the grid's batch axis), so we recover it
//   from the launch geometry. The decode b = blockIdx.z + the batch_stride_o term
//   make B>1 correct.

// Convex-combination combine for ONE output tile.
//   p       : combine kernarg block (scratch pointers, O pointer + strides, G).
//   b       : batch index  (blockIdx.z)
//   h       : head index   (blockIdx.x)
//   m_tile  : M-tile index (blockIdx.y)  -> absolute row R = m_tile*kM0 + row.
__device__ __forceinline__ void combine_split(
    const FmhaFwdCombineParams& p, int b, int h, int m_tile)
{
    constexpr int kD = kHeadDim;   // 64 head-dim columns this kernel handles

    // One row per thread; threads >= kM0 have no row to own and bail.
    const int row = threadIdx.x;
    if (row >= kM0) return;

    // Absolute query row; rows past the real sequence length are padding (the O
    // buffer's tail tile may extend past seqlen_q) and must not be written.
    const int R = m_tile * kM0 + row;
    if (R >= p.seqlen_q) return;

    const int G  = p.num_splits;
    const int Hq = p.nhead_q;
    const int Sq = p.seqlen_q;
    // B is not a struct field; recover it from the grid's batch axis. The g-stride
    // of the split-major scratch is (B*Hq*Sq) elements for LSE / *64 for O.
    const int B  = gridDim.z;

    // Split-major scratch indices (match the split-forward pass EXACTLY):
    //   scratch_o  (g,b,h,R,d) = ((((g*B + b)*Hq + h)*Sq + R)*64 + d
    //   scratch_lse(g,b,h,R)   =  (((g*B + b)*Hq + h)*Sq + R
    // The per-g stride lets us step plane to plane by adding a constant; we keep
    // the explicit form for clarity (this pass is not perf-critical).
    const long bh_row    = (long)((b * Hq + h) * Sq + R);   // (b,h,R) within a plane
    const long g_stride_lse = (long)B * Hq * Sq;            // elements between LSE planes
    const long g_stride_o   = g_stride_lse * kD;            // elements between O planes

    // ---------------------------------------------------------------------
    // num_splits == 1 fast path: a single plane carries the full softmax, so
    // its weight is 1 and the combine is a straight copy. This both (a) avoids
    // the needless exp()/divide and (b) makes the G=1 identity bit-exact: O[d]
    // is o_part[0][d] truncated to bf16 (i.e. bf16_to_float(float_to_bf16(
    // o_part[0][d])); our perm-truncation drops the same low 16 bits).
    // ---------------------------------------------------------------------
    // bf16 truncation selector: pick the HIGH 16 bits of each fp32 (bytes 3,2 and
    // 7,6) — i.e. drop the low mantissa bits == truncation, NOT round-to-nearest.
    // Identical to op_epilog.hpp's kBf16TruncSel and bf16_utils' (u >> 16).
    constexpr unsigned kBf16TruncSel = 0x07060302;

    // Destination O index (bf16 ELEMENTS): same mapping as the epilogue contract,
    //   O index = b*batch_stride_o + h*nhead_stride_o + R*stride_o + d.
    const long o_row_base =
        (long)b * p.batch_stride_o + (long)h * p.nhead_stride_o + (long)R * p.stride_o;
    uint16_t* o_u16 = reinterpret_cast<uint16_t*>(p.o);

    // Optional fp32 precision tap (p.o_fp32 != nullptr): the EXACT fp32 result the
    // bf16 store rounds, in its OWN CONTIGUOUS natural-order layout [B][Hq][Sq][64]
    // (NOT the strided bf16 o_row_base). It lets a caller check the reweight at
    // ~1e-5 to catch reweight-weight bugs the bf16 (~1e-3) store tolerance hides.
    const long of_base = (((long)(b * Hq + h) * Sq + R) * kD);

    if (G == 1) {
        const float* o0 = p.scratch_o + bh_row * kD;   // plane 0, this (b,h,R)
        #pragma unroll
        for (int d = 0; d < kD; ++d) {
            float v = o0[d];
            // Truncate via perm (high half of the dword); store the low 16 bits.
            unsigned packed = __builtin_amdgcn_perm(
                0u, reinterpret_cast<unsigned&>(v), kBf16TruncSel);
            o_u16[o_row_base + d] = (uint16_t)(packed & 0xFFFFu);
        }
        // Optional fp32 tap: with one split the result is plane 0 verbatim — write
        // the un-truncated o0[d] in natural order (guarded by null; off by default).
        if (p.o_fp32) {
            #pragma unroll
            for (int d = 0; d < kD; ++d) p.o_fp32[of_base + d] = o0[d];
        }
        // Optional global LSE: with one split, LSE == the single plane's LSE.
        if (p.lse) {
            float lse0 = p.scratch_lse[bh_row];
            p.lse[(long)b * Hq * Sq + (long)h * Sq + R] = lse0;
        }
        return;
    }

    // ---------------------------------------------------------------------
    // General case: max-subtract reduction over the G planes for this row.
    // ---------------------------------------------------------------------
    // Step 1: global max M of the G LSE scalars.
    float M = -INFINITY;
    for (int g = 0; g < G; ++g) {
        float lse_g = p.scratch_lse[bh_row + (long)g * g_stride_lse];
        if (lse_g > M) M = lse_g;
    }

    // Output accumulator for this row's 64 head-dim columns (fp32).
    float acc[kD];
    #pragma unroll
    for (int d = 0; d < kD; ++d) acc[d] = 0.0f;

    // All ranges empty/masked -> no mass; O stays zero (no NaN). Falls through to
    // the store below, writing zeros.
    if (M != -INFINITY) {
        // Step 2: unnormalized weights w_g = exp(lse_g - M) and their sum.
        // denom > 0 always (the plane with lse_g == M contributes exp(0) = 1),
        // but we still guard the reciprocal.
        // Accumulate the denominator in DOUBLE precision. With wide LSE spreads
        // and large G, fp32 denom drift would otherwise exceed the 1e-3 bf16-store
        // tolerance. Use the precise expf (NOT the fast __expf intrinsic) so the
        // weights stay accurate to fp32 ULPs.
        double denom = 0.0;
        // Two passes keep the register footprint tiny (no per-g weight array):
        // first sum the denominator, then re-walk the planes accumulating O.
        for (int g = 0; g < G; ++g) {
            float lse_g = p.scratch_lse[bh_row + (long)g * g_stride_lse];
            float wg = (lse_g == -INFINITY) ? 0.0f : expf(lse_g - M);
            denom += wg;
        }
        float inv = (denom > 0.0) ? (float)(1.0 / denom) : 0.0f;

        for (int g = 0; g < G; ++g) {
            float lse_g = p.scratch_lse[bh_row + (long)g * g_stride_lse];
            if (lse_g == -INFINITY) continue;        // weight 0: skip the plane
            float wg = expf(lse_g - M) * inv;        // normalized weight, Σ=1
            // Plane g's (b,h,R) row base, then index d within the natural-order plane.
            const float* og_row = p.scratch_o + (bh_row * kD) + (long)g * g_stride_o;
            #pragma unroll
            for (int d = 0; d < kD; ++d) acc[d] += wg * og_row[d];
        }
    }

    // Step 3: bf16-truncate the 64 fp32 results and store to O at natural columns.
    #pragma unroll
    for (int d = 0; d < kD; ++d) {
        float v = acc[d];
        unsigned packed = __builtin_amdgcn_perm(
            0u, reinterpret_cast<unsigned&>(v), kBf16TruncSel);
        o_u16[o_row_base + d] = (uint16_t)(packed & 0xFFFFu);
    }

    // Optional fp32 tap: write the EXACT convex-combination accumulator acc[] in
    // natural order BEFORE the bf16 truncation above — this is the whole point of
    // the tap (it exposes reweight-weight errors the bf16 store would round away).
    // Guarded by null; the default-null callers leave the bf16 path byte-identical.
    if (p.o_fp32) {
        #pragma unroll
        for (int d = 0; d < kD; ++d) p.o_fp32[of_base + d] = acc[d];
    }

    // Optional global LSE: L* = M + ln(denom) in natural units. The split-K
    // callers pass params.lse == nullptr, so this path is currently unexercised
    // but implemented for completeness and guarded so a null pointer never faults.
    if (p.lse) {
        // Recompute denom cheaply (kept out of the hot path above to avoid a live
        // register across the O accumulation); M is finite here unless all-empty.
        if (M == -INFINITY) {
            p.lse[(long)b * Hq * Sq + (long)h * Sq + R] = -INFINITY;
        } else {
            double denom = 0.0;
            for (int g = 0; g < G; ++g) {
                float lse_g = p.scratch_lse[bh_row + (long)g * g_stride_lse];
                denom += (lse_g == -INFINITY) ? 0.0f : expf(lse_g - M);
            }
            p.lse[(long)b * Hq * Sq + (long)h * Sq + R] = M + logf((float)denom);
        }
    }
}
