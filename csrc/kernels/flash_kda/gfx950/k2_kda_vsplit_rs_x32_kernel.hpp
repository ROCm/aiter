#pragma once

#include "../k2_kda_vsplit_rs_kernel.hpp"
#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

// gfx950-private register-state contraction.  The state mapping is unchanged:
// adjacent K16 fragments are packed with the same permutation as the LDS A
// operand, replacing eight x16 MFMAs with four native x32 MFMAs for K=128.
struct RegBX32 {
    template <int LD, int NKB>
    static __device__ __forceinline__ f32x4 run(
            const __bf16* __restrict__ a,
            const float (&state)[NKB][4],
            int lane) {
        return gemm_regb_even_x32<LD, NKB>(a, state, lane);
    }
};

// Publish each kr K16 slice as one contiguous [C,K] tile.  The carry then forms
// both MFMA operands with native transpose reads: eight tr-reads replace the 32
// scalar LDS reads and their fragment-building permutes in each chunk.
struct TiledKrCarryX16 {
    template <int C, int D, int RW>
    static __device__ __forceinline__ void store(
            __bf16* __restrict__ kr,
            const bf16x8 (&staged)[RW],
            int lane) {
        static_assert(C == 16 && D % C == 0,
                      "tiled kr carry requires K16 tiles");
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int source_element = (lane + j * 64) * 8;
            const int c = source_element / D;
            const int k = source_element - c * D;
            const int kt = k / C;
            const int ki = k - kt * C;
            __bf16* destination = kr + kt * C * C + c * C + ki;
            *reinterpret_cast<bf16x8*>(destination) = staged[j];
        }
    }

    template <int C, int D, int BV>
    static __device__ __forceinline__ f32x4 run(
            const __bf16* __restrict__ kr,
            const __bf16* __restrict__ umat,
            int kt,
            int vt,
            int lane) {
        static_assert(C == 16 && D % C == 0 && BV == C,
                      "tiled kr carry is specialized for BV16");
        (void)vt;
        const bf16x4 a = ds_read_tr16(kr + kt * C * C, lane);
        const bf16x4 b = ds_read_tr16(umat, lane);
        const f32x4 zero = {0.f, 0.f, 0.f, 0.f};
        return mfma_bf16(a, b, zero);
    }
};

}  // namespace flashkda_hip::gfx950
