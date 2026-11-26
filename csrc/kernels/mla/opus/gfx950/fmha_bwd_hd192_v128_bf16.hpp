// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device code of the OPUS attention backward on gfx950 (bf16, D_QK = 192, D_V = 128): the
// pre-pass (the output gate, D) and the dK / dV and dQ passes. Built into a code object with the
// amdgpu-named-reg-pin-std toolchain (fmha_bwd_hd192_v128_bf16.cu, build_co.sh); the host side
// (fmha_bwd_bf16_opus_kernels.cu) knows only fmha_bwd_hd192_v128_bf16_opus_defs.h.
//
// The loops' schedule is laid out by hand, a fragment's work to a line and its tables aligned
// by column, so clang-format leaves the file as it is.
// clang-format off
#pragma once

#include <hip/hip_runtime.h>

#include "fmha_bwd_hd192_v128_bf16_opus_defs.h"

#include <opus/opus.hpp>

namespace fmha_bwd_d192_v128 {

using bf16_t = opus::bf16_t;
template <int N>
using bvec    = opus::vector_t<bf16_t, N>;
using f32x4_t = opus::vector_t<float, 4>;

// __syncthreads, which the device-only build's HIP headers leave out: no memory access crosses it in the compiler.
__device__ inline void block_sync()
{
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_s_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}
using u32x2_t = opus::vector_t<opus::u32_t, 2>;
using u32x4_t = opus::vector_t<opus::u32_t, 4>;
using i32x4_t = opus::vector_t<int, 4>;

constexpr float LOG2_E = 1.44269504089f;  // exps are exp2: the scale and LSE carry log2 e

// This wave's global loads and DMA rounds are in (s_waitcnt vmcnt(0), as an instruction the compiler sees: it
// then adds no wait of its own for them). The signal fence: no memory access crosses it in the compiler.
__device__ inline void wait_loads()
{
    opus::s_waitcnt_vmcnt(opus::number<0>{});
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}
// The same in one instruction with a wait for this wave's LDS reads but the last N.
template <int N>
__device__ inline void wait_loads_reads()
{
    opus::s_waitcnt(opus::number<0>{}, opus::number<N>{});
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// sigmoid(g), recomputed in fp32 from the logits the op is given: no s is kept from the forward,
// and 1 - s keeps its precision where a stored bf16 s would lose it.
__device__ inline float sigmoid(float g)
{
    return __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-LOG2_E * g));
}

// One transposed operand read for a 16x16x32 MFMA from a row-major LDS tile whose rows run
// along the GEMM's K. ds_read_b64_tr_b16: each 16-lane group addresses a 4 x 16 block (lane
// i: row i / 4, columns 4 (i % 4) .. + 3) and lane i receives column i, 4 rows deep.
// This is the compiler builtin, not opus' inline-asm smem::tr_load: the compiler cannot see
// inline-asm LDS reads, so it neither waits for them nor keeps them under the 15 outstanding
// LDS operations lgkmcnt can count.
__device__ inline u32x2_t tr_read(const opus::smem<bf16_t>& s, int elem_off)
{
    using i16x4_t = opus::vector_t<short, 4>;
    // Through a __restrict__ pointer: inlined, the read carries an alias scope, so the compiler's DMA tracking
    // does not make it wait for the DMA rounds in flight (as it does not for the plain LDS loads). The kernel
    // waits for its DMA itself.
    auto rd = [](OPUS_LDS_ADDR i16x4_t* __restrict__ q) __attribute__((always_inline)) {
        return __builtin_amdgcn_ds_read_tr16_b64_v4i16(q);
    };
    return __builtin_bit_cast(u32x2_t, rd(reinterpret_cast<OPUS_LDS_ADDR i16x4_t*>(s.ptr + elem_off * static_cast<int>(sizeof(bf16_t)))));
}

// This lane's part of the address of the transposed reads that make the B fragment of a 16x16x32
// MFMA from a tile with rows of LD elements, in the K order of two swap_ab 16x16 C tiles (which
// are then that MFMA's A operand as they are): slot 4 h + e of lane group g is row 16 h + 4 g + e.
// The reads are at this offset plus 16 n for N tile n and 16 LD h for half h.
template <int LD>
__device__ inline int tr_c_order_lane(int lane)
{
    const int g = lane / 16, i = lane % 16;
    return (g * 4 + i / 4) * LD + (i % 4) * 4;
}

// 8 bf16 at element 8 i of a fragment, as the 128-bit MFMA operand.
template <typename V>
__device__ inline u32x4_t frag8(const V& v, int i)
{
    opus::vector_t<bf16_t, 8> r;
#pragma unroll
    for(int e = 0; e < 8; e++)
        r[e] = v[8 * i + e];
    return __builtin_bit_cast(u32x4_t, r);
}

// The dK, dV and dQ accumulators stay in AGPRs: the module is built with -amdgpu-mfma-vgpr-form,
// so the compiler's own MFMAs accumulate in VGPRs and the AGPR file holds only these long-lived
// tiles, each at a fixed place (pin_agpr) from the zeros to the epilogue.

// The buffer descriptor of a global -> LDS copy (buffer_load_dwordx4 ... lds): lane l's 16 bytes
// at its byte offset of the buffer (past the descriptor's size: zeros) land at LDS byte m0 + 16 l.
// The copies are llvm.amdgcn.raw.buffer.load.lds (dma_round); the compiler waits for them only
// before LDS reads it cannot tell apart from them (the reads here carry TBAA or an alias scope).
// The kernel waits vmcnt itself before anyone reads the data, which s_barrier alone does not do.
__device__ inline i32x4_t dma_rsrc(const void* p, unsigned int bytes)
{
    const unsigned long long a = reinterpret_cast<unsigned long long>(p);
    return i32x4_t{__builtin_amdgcn_readfirstlane(static_cast<int>(a)),
                   __builtin_amdgcn_readfirstlane(static_cast<int>(a >> 32) & 0xffff),
                   __builtin_amdgcn_readfirstlane(static_cast<int>(bytes)),
                   opus::buffer_default_config()};
}
__device__ inline unsigned int lds_addr(const void* p)
{
    return static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(p));
}

// A value through a named register group: its definition lands in a[REG ..] / v[REG ..].
template <int REG, class V>
__device__ inline V pin_agpr(V v)
{
    __attribute__((amdgpu_pin_agpr(REG))) V r = v;
    return r;
}
template <int REG, class V>
__device__ inline V pin_vgpr(V v)
{
    __attribute__((amdgpu_pin_vgpr(REG))) V r = v;
    return r;
}

// One global -> LDS DMA round (opus' async_load): lane l's `bytes` at byte offset os of the buffer r (four
// words) land at LDS byte lds + bytes l. The intrinsic by its name, on the descriptor's four words: the builtin takes
// a buffer resource, and making one (make_buffer_rsrc) masks the address's high half, an s_and more in the loop
// for every step of a descriptor (x_adv).
__device__ void raw_buffer_load_lds(i32x4_t r, __attribute__((address_space(3))) void* lds, int size, int vos, int sos, int ios,
                                    int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
template <int BYTES>
__device__ inline void dma_round(const i32x4_t& r, unsigned int lds, int os)
{
    raw_buffer_load_lds(r, reinterpret_cast<__attribute__((address_space(3))) void*>(lds), BYTES, os, 0, 0, 0);
}

// Co-execution scheduling (sched_group_barrier), as in the forward kernels. One wave per SIMD: an MFMA leaves three
// places behind it; the first is free only for a scalar or a wait, a VALU or an LDS read takes one of the other
// two (an exp both, a DMA round the three). A fragment (an N tile's K step: one MFMA per M tile) is written as its
// MFMAs, then what rides behind them in the order it is to issue, then one sched_mfma per MFMA and a fence. A group
// takes the code's first instructions of its class that no group before it took, and the groups issue in their
// order, so each group is given exactly what the code has for it.
constexpr int VALU_MASK = 0x2, SALU_MASK = 0x4, MFMA_MASK = 0x8, DS_READ_MASK = 0x100, EXP_MASK = 0x400, DMA_MASK = 0x800;
// 1 MFMA and, behind it in this order, SALU scalar instructions (an anchored scalar is two: the op and its anchor;
// a wait is one), READ LDS reads, VALU VALUs, EXP exps, DMA DMA rounds (the round's LDS address into m0 is a scalar
// of SALU).
template <int SALU, int VALU, int EXP = 0, int READ = 0, int DMA = 0>
__device__ inline void sched_mfma()
{
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, 0);
    if constexpr(SALU > 0)
        __builtin_amdgcn_sched_group_barrier(SALU_MASK, SALU, 0);
    if constexpr(READ > 0)
        __builtin_amdgcn_sched_group_barrier(DS_READ_MASK, READ, 0);
    if constexpr(VALU > 0)
        __builtin_amdgcn_sched_group_barrier(VALU_MASK, VALU, 0);
    if constexpr(EXP > 0)
        __builtin_amdgcn_sched_group_barrier(EXP_MASK, EXP, 0);
    if constexpr(DMA > 0)
        __builtin_amdgcn_sched_group_barrier(DMA_MASK, DMA, 0);
}

} // namespace fmha_bwd_d192_v128

// One row of D_V = 128 bf16 per 16 lanes (8 per lane, one 16-byte access each). GATE: the output
// is gated, and the pre-pass makes dO, dgate and D; without it, D alone.
template <bool GATE>
__device__ inline void fmha_bwd_v128_pre_body(const fmha_bwd_v128_pre_kargs& k)
{
    using bf16_t = opus::bf16_t;
    constexpr int D_V   = 128;
    constexpr int VEC   = 8;
    constexpr int LANES = D_V / VEC;
    constexpr int ROWS  = FMHA_BWD_V128_PRE_BLOCK / LANES;

    const int tid = static_cast<int>(opus::thread_id_x());
    const long long row = static_cast<long long>(opus::block_id_x()) * ROWS + tid / LANES;
    const int col = (tid % LANES) * VEC;
    const bool valid = row < static_cast<long long>(k.total) * k.H;
    const int t = static_cast<int>(valid ? row / k.H : 0);
    const int h = static_cast<int>(valid ? row % k.H : 0);

    float acc = 0.0f;
    if(valid)
    {
        auto at = [&](const void* p, int sn, int sh) {
            return reinterpret_cast<const bf16_t*>(p) + static_cast<long long>(t) * sn +
                   static_cast<long long>(h) * sh + col;
        };
        using vec_t = opus::vector_t<bf16_t, VEC>;
        auto dout = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_dout, k.stride_dout_n, k.stride_dout_h)));
        auto out  = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_out, k.stride_out_n, k.stride_out_h)));
        if constexpr(!GATE)
        {
            // dO is dout itself.
            for(int i = 0; i < VEC; i++)
            {
                const float d_o = dout[i] * out[i];
                acc += d_o;
            }
        }
        else
        {
            auto gate = opus::cast<float>(*reinterpret_cast<const vec_t*>(at(k.ptr_gate, k.stride_gate_n, k.stride_gate_h)));
            opus::vector_t<float, VEC> v_do, v_dgate;
            for(int i = 0; i < VEC; i++)
            {
                const float s    = fmha_bwd_d192_v128::sigmoid(gate[i]);
                const float d_o  = dout[i] * out[i];
                v_do[i]    = dout[i] * s;
                v_dgate[i] = d_o * (1.0f - s);
                acc += d_o;
            }
            auto wr = [&](void* p, int sn, int sh) {
                return reinterpret_cast<bf16_t*>(p) + static_cast<long long>(t) * sn +
                       static_cast<long long>(h) * sh + col;
            };
            *reinterpret_cast<vec_t*>(wr(k.ptr_do, k.stride_do_n, k.stride_do_h)) = opus::cast<bf16_t>(v_do);
            *reinterpret_cast<vec_t*>(wr(k.ptr_dgate, k.stride_dgate_n, k.stride_dgate_h)) =
                opus::cast<bf16_t>(v_dgate);
        }
    }
    // A row's 16 lanes are aligned within the wave, so xor offsets below 16 stay in the row.
    const int lane = tid % opus::get_warp_size();
    for(int m = LANES / 2; m > 0; m >>= 1)
        acc += opus::shfl(acc, lane ^ m);
    if(valid && tid % LANES == 0)
        k.ptr_d[static_cast<long long>(h) * k.stride_d_h + t] = acc;
}

// The dK / dV pass's loop: 40 fragments in the loop's order, S^T's 12 (fragment f: N tile f / 6, K step f % 6), dP^T's
// 8 (N tile f / 4, K step f % 4), dK's 12 and dV's 8 (an N tile each), every fragment 3 MFMAs, one per M tile. The
// rules below say where the loop's work rides behind them, each kind of work in an order of its own:
//   scalars  the loop's scalars, in the order of C_... (S^T only)
//   lses     LSE log2 e of query index x (N tile x / 4): behind the wait that has its LSE in
//   pds      dS of the tile before to bf16, two elements each: the A operand of the dK block (S^T only)
//   fmas     S^T scale - LSE           exps     its exp, P
//   muls     P (dP^T - D), dS in fp32: dP^T's accumulators start at -D
//   pps      P to bf16, two elements each: the A operand of the dV block
//   negs     -D of the next tile's query index x: behind the wait that has its D in
//   addrs    an LDS address a tile on: LSE's and D's, Q^T's from Q's, then Q's, dO's, dO^T's
// Element e of a wave's 24 S^T / dP^T values per lane is C tile (M tile e / 8, N tile e / 4 % 2); its query's LSE
// and D have index (e / 4 % 2) 4 + e % 4. fma, exp, mul and pp take the elements N tile by N tile (elem).
// An MFMA leaves three places behind it, the first free only for a scalar or a wait, the others for a VALU or an
// LDS read each (an exp takes both, a DMA round the three). A fragment's last MFMA takes the wait and the fragment's
// LDS reads; every other one has the same share, two VALU and one exp (the first of dP^T's fragments 1 and 5 a read
// and one VALU; the first of S^T's first four, with a DMA descriptor's step, nothing: a VALU behind the step's two
// scalars costs a cycle; the second of S^T's first seven the DMA round). The shares are filled in the code's order,
// so a fragment is of exps, one an MFMA, or of VALU: with both, its first MFMAs would get an exp and VALU, more
// than their three places. The work keeps its order: an fma behind the last MFMA of its C tile in S^T and a mul
// behind that in dP^T by four MFMAs or more (a VALU reads an MFMA's result); an fma in a fragment behind its LSE
// log2 e's, an exp behind its fma's, a mul and a pp behind their exps' (a value's register is written at the end
// of the fragment that makes it); a pp in front of dV, an address behind the last LDS read at the old one.
namespace fmha_bwd_d192_v128_dkdv_sched {
// The loop's scalars, in the order the S^T block takes them.
enum : signed char { C_RC_INC, C_RC_AND, C_DN_SUB, C_DN_SHR, C_DN_AND, C_LBC, C_LBP, C_DLT, C_RARE, C_Z, C_RA, C_FD, C_N };
// The blocks' fragments and the loop's index of each block's first; the M tiles and the elements of a wave.
constexpr int NF_S = 12, NF_P = 8, NF_K = 12, NF_V = 8, I_P = NF_S, I_K = I_P + NF_P, I_V = I_K + NF_K, NL = I_V + NF_V;
constexpr int MR = 3, NE = 8 * MR;
// The items first .. first + n - 1 of a kind of work.
struct span
{
    int first, n;
};
constexpr int clamp(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
constexpr int none(int)
{
    return 0;
}
// The i-th of a wave's 24 elements N tile by N tile.
constexpr int elem(int i)
{
    return 8 * (i / 4 % 3) + 4 * (i / 12) + i % 4;
}
// Bit e set if element e's exp is among the first n (those of the S^T block: the mask of a tile not seen in full is
// applied behind it, and those elements take P = 0, the others an S^T that exps to 0).
constexpr unsigned int exp_done(int n)
{
    unsigned int m = 0;
    for(int i = 0; i < n; i++)
        m |= 1u << elem(i);
    return m;
}
// The DMA descriptor that goes a tile on behind each of the S^T block's first four fragments.
inline constexpr signed char S_ADV[4] = {1, 2, 3, 0};

// S^T's first four fragments step a DMA descriptor behind their first MFMA, its first seven issue a DMA round behind
// their second.
constexpr bool has_step(int i)
{
    return i < 4;
}
constexpr bool has_dma(int i)
{
    return i < 7;
}
// The rings of the B operands: a block's fragment f is in entry f % 3 of its ring (bq, bdo, tb), and behind
// fragment f's last MFMA that entry is refilled with the fragment three ahead, as the MLA kernels refill their K
// ring behind a step's MFMAs; a block's last three fragments read the next block's first three instead (dV's the
// next tile's S^T's). S^T's and dV's last fragment, in front of a branch, reads nothing: the one before it reads
// two. Behind S^T's first two fragments LSE's two N tiles are read too, and behind dP^T's fragments 1 and 2 the
// next tile's D's.
// dP^T's fragments 0 and 4 start from -D, whose registers are free behind their last MFMA: a read there could take
// them while that MFMA still reads them, and the compiler would wait. So no read stands there: their entries'
// refill goes behind the next fragment's first MFMA.
// reads(i): the LDS reads behind the last MFMA of the loop's fragment i (the one before its first is its last); an
// N tile of Q^T or dO^T is two. reads_head(i): those behind its first MFMA.
constexpr int reads(int i)
{
    i = (i + NL) % NL;
    // S^T: Q's fragment three ahead and, behind the first two, an N tile of LSE; its last three read dO's first
    // three: one, two, none (in front of the branch).
    if(i < I_P)
    {
        if(i + 3 < NF_S)
            return i < 2 ? 2 : 1;
        return i == NF_S - 3 ? 1 : (i == NF_S - 2 ? 2 : 0);
    }
    // dP^T: dO's fragment three ahead (none behind fragments 0 and 4: reads_head) and, behind fragments 1 and 2, an
    // N tile of D; its last three read Q^T's first three N tiles.
    if(i < I_K)
    {
        const int f = i - I_P;
        if(f + 3 < NF_P)
            return (f % 4 == 0 ? 0 : 1) + (f == 1 || f == 2 ? 1 : 0);
        return 2;
    }
    // dK: Q^T's N tile three ahead; its last three read dO^T's first three.
    if(i < I_V)
        return 2;
    // dV: dO^T's N tile three ahead; its last three read the next tile's first three fragments of Q: one, two, none
    // (in front of the branch).
    const int g = i - I_V;
    if(g + 3 < NF_V)
        return 2;
    return g == NF_V - 3 ? 1 : (g == NF_V - 2 ? 2 : 0);
}
constexpr int reads_head(int i)
{
    return i >= I_P && i < I_K && (i - I_P) % 4 == 1 ? 1 : 0;
}
// The loop's scalars: one behind each of S^T's fragments 4, 5 and 6, then two a fragment.
constexpr span scalars(int i)
{
    return i >= 4 && i < 7 ? span{i - 4, 1} : i >= 7 && i < I_P ? span{2 * i - 11, clamp(C_N - (2 * i - 11), 0, 2)} : span{0, 0};
}
// The MFMAs of a fragment that have the one share: all but the last, the step's, the DMA round's and the one with
// a read.
constexpr int shares(int i)
{
    return MR - 1 - has_step(i) - has_dma(i) - reads_head(i);
}
// The fragments of exps: S^T's last, dP^T's fragments 3, 6 and 7, dK's first eight.
constexpr bool of_exps(int i)
{
    return i == I_P - 1 || i == I_P + 3 || i == I_P + 6 || i == I_P + 7 || (i >= I_K && i < I_K + 8);
}
// The VALU a fragment has room for: none if it is of exps; else two behind each MFMA with the one share, one beside
// a read behind the first MFMA and, with LAST, what the reads and a scalar leave of the two places behind the last
// MFMA (none in front of a branch).
constexpr int room(int i, bool last = true)
{
    if(of_exps(i))
        return 0;
    const bool branch = i == I_P - 1 || i == NL - 1;
    const int s_last = scalars(i).n > MR - 1 - has_step(i) - has_dma(i) ? 1 : 0;
    return 2 * shares(i) + reads_head(i) + (last && !branch ? 2 - reads(i) - s_last : 0);
}
// Of the n items of a kind that ride from fragment a on, each fragment taking as many as it has room for beside
// the USED ones of the kinds in front of it: those behind fragment i.
template<class U>
constexpr span fill(int a, int n, int i, U used, bool last = true)
{
    int first = 0;
    for(int j = a; j < i && first < n; j++)
        first += clamp(room(j, last) - used(j), 0, n - first);
    return {first, i < a ? 0 : clamp(room(i, last) - used(i), 0, n - first)};
}
// LSE log2 e, index x. N tile 0's four: one behind each of S^T's fragments 2 .. 5 (in at the wait of fragment 2).
// N tile 1's: two behind S^T's fragment 10, two behind dP^T's first.
constexpr span lses(int i)
{
    return i >= 2 && i < 6 ? span{i - 2, 1} : i == 10 ? span{4, 2} : i == I_P ? span{6, 2} : span{0, 0};
}
// dS of the tile before to bf16, pair j: from S^T's fragment 4 on, two behind each MFMA with the one share.
constexpr span pds(int i)
{
    return fill(4, NE / 2, i, none, false);
}
// S^T scale - LSE. N tile 0's: from S^T's fragment 6 on (its last MFMA: four behind the first C tile's last), in
// what the pairs and LSE log2 e leave. N tile 1's: two behind dP^T's first fragment (the ones whose LSE log2 e
// stands in S^T), the others from its second on.
constexpr span fmas(int i)
{
    const span a = fill(6, NE / 2, i, [](int j) { return pds(j).n + lses(j).n; });
    const span b = fill(I_P + 1, NE / 2 - 2, i, none);
    return a.n ? a : i == I_P ? span{NE / 2, 2} : span{NE / 2 + 2 + b.first, b.n};
}
// The exps: one behind each MFMA with the one share of the fragments of exps.
constexpr span exps(int i)
{
    int first = 0;
    for(int j = 0; j < i; j++)
        first += of_exps(j) ? shares(j) : 0;
    return {first, of_exps(i) ? clamp(NE - first, 0, shares(i)) : 0};
}
// dS = P (dP^T - D). Those of N tile 0's first M tile: behind dP^T's fragments 4 and 5, in what the fma leave (the
// last MFMA of fragment 4 is to have a VALU: fragment 5's read behind its first MFMA then stands far enough behind
// the MFMAs that start from -D). The others: from dK's last fragment on.
constexpr span muls(int i)
{
    const span a = fill(I_P + 4, 4, i, [](int j) { return fmas(j).n; });
    const span b = fill(I_K + NF_K - 1, NE - 4, i, none);
    return a.n ? a : span{4 + b.first, b.n};
}
// P to bf16, pair j N tile by N tile: from dK's fragment 8 on (behind its eight fragments of exps).
constexpr span pps(int i)
{
    return fill(I_K + 8, NE / 2, i, none);
}
// -D of the next tile, index x: two behind dP^T's fragment 5, beside its dS; the others from dV's fragment 4 on.
constexpr span negs(int i)
{
    const span a = fill(I_P + 5, 2, i, [](int j) { return muls(j).n; });
    const span b = fill(I_V + 4, 6, i, none);
    return a.n ? a : span{2 + b.first, b.n};
}
// The LDS addresses a tile on. LSE's and D's: behind dP^T's first fragment (between LSE's last read and D's first).
// The others: from dV's fragment 5 on (behind dO^T's last read), in what -D leaves.
constexpr span addrs(int i)
{
    const span b = fill(I_V + 5, 4, i, [](int j) { return negs(j).n; });
    return i == I_P ? span{0, 1} : span{1 + b.first, b.n};
}
// All the loop's items of a kind.
template<class K>
constexpr int total(K kind)
{
    int n = 0;
    for(int i = 0; i < NL; i++)
        n += kind(i).n;
    return n;
}
}  // namespace fmha_bwd_d192_v128_dkdv_sched

// dK / dV pass of the 4-wave backward. Each wave runs at one per SIMD so that its 48 keys' K and V (A of
// S^T and dP^T) stay in VGPRs and dK and dV in AGPRs: every LDS fragment of Q and dO feeds
// three M tiles. GEMMs are per wave, C = A * B^T, all 16x16x32 MFMAs: S^T and dP^T over d; dV
// and dK over the tile's 32 queries, with P^T and dS^T (the C fragments of S^T and dP^T, two N
// tiles per M tile) as their A operands and transposed reads of the dO and Q tiles as B. Query
// tiles go global -> LDS by DMA two tiles ahead into a ring of four buffers, one barrier per tile.
__device__ __attribute__((always_inline)) inline void fmha_bwd_d192_v128_dkdv_body(
    const fmha_bwd_d192_v128_kargs& k, fmha_bwd_d192_v128::bf16_t* smem, const int lane, const int w, const int h, const int hk, const int j,
    unsigned long long bnd_q, unsigned long long bnd_k, fmha_bwd_d192_v128::f32x4_t (&z)[4][12])
{
    using namespace fmha_bwd_d192_v128_dkdv;
    using fmha_bwd_d192_v128::bf16_t;
    using fmha_bwd_d192_v128::bvec;
    using fmha_bwd_d192_v128::f32x4_t;
    using opus::number;
    using opus::operator""_I;
    using vec8_t = bvec<8>;
    using fmha_bwd_d192_v128::LOG2_E;
    constexpr int MR       = WR / 16;  // M tiles per wave
    constexpr int NT       = BQ / 16;  // N tiles of S^T / dP^T
    // dK's accumulators, [M tile][N tile] in the swap_ab C layout, in AGPRs: the kernel's zeros.
    constexpr int NV = D_V / 16, NR = D_ROT / 16;
    static_assert(MR == 3 && NV + NR == 12, "the kernel's zeroed tiles");
    // The AGPR file: dK's tile (m, n) at a[4 (12 m + n)], where the kernel put its zeros; dV's behind
    // dK's 36 tiles.
    constexpr int A_DK = 0, A_DV = 4 * MR * (NV + NR);
    static_assert(A_DV + 4 * MR * NV <= 256, "the AGPR file");
    f32x4_t acc_dkn[MR][NV], acc_dkr[MR][NR];
#pragma unroll
    for(int m = 0; m < MR; m++)
    {
#pragma unroll
        for(int n = 0; n < NV; n++)
            acc_dkn[m][n] = z[m][n];
#pragma unroll
        for(int n = 0; n < NR; n++)
            acc_dkr[m][n] = z[m][NV + n];
    }

    // The tensors' bases at this head.
    const bf16_t* q_h  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(h) * k.stride_q_h;
    const bf16_t* do_h = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(h) * k.stride_do_h;
    const bf16_t* kn_h = reinterpret_cast<const bf16_t*>(k.ptr_k) + static_cast<long long>(hk) * k.stride_k_h;
    const bf16_t* v_h  = reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(hk) * k.stride_v_h;
    const float* ld_h  = (w % 2 == 0 ? k.ptr_lse : k.ptr_d) + static_cast<long long>(h) * k.stride_lse_h;

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    static_assert(D_V == D_NOPE, "dV and the nope part of dK share one tile shape");
    static_assert(BQ == 32, "P^T / dS^T are the A operand of one 16x16x32 K step");
    auto mma_s  = make_mma(number<MR>{}, number<NT>{}, number<D_QK / 32>{});    // S^T  [48 x 32]
    auto mma_kn = make_mma(number<MR>{}, number<NT>{}, number<D_NOPE / 32>{});  // A of S^T, nope part
    auto mma_kr = make_mma(number<MR>{}, number<NT>{}, number<D_ROT / 32>{});   // A of S^T, rope part
    auto mma_dp = make_mma(number<MR>{}, number<NT>{}, number<D_V / 32>{});     // dP^T [48 x 32]
    auto mma1   = make_mma(number<1>{}, number<1>{}, number<1>{});
    const auto c_ab = opus::make_tuple(0_I, lane % mma_s.grpm_a, 0_I, lane / mma_s.grpm_a);

    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };

    // C element e = (m NT + tn) 4 + p of S^T / dP^T is key 16 m + lane % 16 of the wave and
    // query 16 tn + 4 (lane / 16) + p of the tile.
    constexpr int NS = decltype(mma_s)::tile_c_len;
    static_assert(NS == MR * NT * 4);
    constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KS_R = D_ROT / 32;
    const int c_q = 4 * (lane / 16);

    // One wave per SIMD: an MFMA leaves three places behind it, the first free only for a scalar or a wait, the
    // others for an LDS read or a VALU each (an exp takes both, a DMA round the three). An iteration (S^T tile ts)
    // is 40 fragments of 3 MFMAs, one per M tile; what rides behind each is in fmha_bwd_d192_v128_dkdv_sched:
    //   barrier               every wave is past its reads of tile ts - 2, tile ts + 1 is whole
    //   S^T(ts)      36 MFMA  the 7 DMA rounds of tile ts + 2 into tile ts - 2's buffer, LSE and LSE log2 e, dS of
    //                         tile ts - 1 to bf16, the loop's scalars, S^T scale - LSE of the C tiles done mid
    //                         block and their first exps
    //   dP^T(ts)     24 MFMA  the other S^T scale - LSE, exps, the next tile's D, the first of dS = P (dP^T - D)
    //   dK(ts - 1)   36 MFMA  the other exps, P to bf16 for dV, dS
    //   dV(ts)       24 MFMA  the last of dS, -D of the next tile, the LDS addresses a tile on
    // S^T and dP^T go N tile by N tile (a B fragment serves three consecutive MFMAs, one per M tile), so half their
    // C tiles are done mid block. dK runs a tile behind so that the VALU work between S^T and dS has the MFMAs of
    // dP^T, dK and dV to go behind. No MFMA waits for LDS: the B operands are in rings of three, and behind a
    // fragment's last MFMA its entry is refilled with the fragment three ahead. dS = P (dP^T - D): dP^T's
    // accumulators start at -D (D is read and negated a tile ahead). Query tiles (Q, dO, LSE, D) go global -> LDS
    // by DMA into NBUF buffers.
    constexpr int XB_Q = OFF_Q, XB_DO = OFF_DO, XB_LSE = OFF_LSE, XB_D = OFF_D, XB_SZ = SZ_BUF, XNB = NBUF;
    static_assert(XNB * XB_SZ <= SMEM_ELEMS, "four tile buffers");
    // A scalar here: the op that makes it stays in front of this place.
    auto here   = [](auto& x) { asm volatile("" : "+s"(x)); };
    auto fence  = [] { __builtin_amdgcn_sched_barrier(0); };
    float sc2   = k.softmax_scale * LOG2_E;
    // A wait for this wave's LDS reads but the last n, a fence behind: in front of the loop and of the tail.
    auto lds_wait = [&](auto n) { opus::s_waitcnt_lgkmcnt(n); fence(); };
    auto lds_of = [&](int b) { return fmha_bwd_d192_v128::lds_addr(smem + b); };

    // DMA: a load is 64 lanes x 16 B from a tile laid out with the row padding, to a wave-uniform
    // LDS address: load jj of a tensor covers its 16-byte slots 64 jj .. + 63, slot s being row
    // s / QS, chunk s % QS of Q (s / DS, s % DS of dO); padding chunks and rows past seqlen_q read
    // out of bounds, as 0. In round xr = 0 .. 5 wave w issues the tile's load w + NW xr (13 of Q then 9
    // of dO; a load past them repeats the wave's last one) and, as round 6, LSE (even w) or D.
    // Rounds 0 .. 2 load Q, 4 and 5 dO, round 3 Q's last load for wave 0 and dO for the others:
    // four descriptors, each pointing at the tile to load (base; size a full tile's, or what
    // x_sizes sets for the sequence's last, partial tile and the tiles past it).
    constexpr int XR = 7, J_Q = 13, J_D = 9;
    constexpr int QS = LD_Q * 2 / 16, DS = LD_DO * 2 / 16;  // 16-byte slots per padded row
    constexpr int OOB = 1 << 30;                            // byte offset past any tile
    static_assert(BQ * QS == J_Q * 64 && BQ * DS == J_D * 64 && NW == 4, "the rounds below");
    // The lanes' byte offsets of load jj (jj wave-uniform; the divisors are constants).
    auto os_q = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % QS;
        const int in = -static_cast<int>(c < D_QK / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / QS * (k.stride_q_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_d = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % DS;
        const int in = -static_cast<int>(c < D_V / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / DS * (k.stride_do_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    const bool r3q = w == 0;
    const int j5   = w < 2 ? w + 7 : w + 3;
    // readfirstlane: the compiler makes these selects in VALU, with the lanes' offsets'.
    auto u_lds = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    int x_lds[XR], x_os[XR];
#pragma unroll
    for(int xr = 0; xr < 3; xr++)
    {
        x_lds[xr] = u_lds(XB_Q * 2 + (w + NW * xr) * 1024);
        x_os[xr]  = os_q(w + NW * xr);
    }
    x_lds[3] = u_lds(r3q ? XB_Q * 2 + 12 * 1024 : XB_DO * 2 + (w - 1) * 1024);
    x_os[3]  = r3q ? os_q(12) : os_d(w - 1);
    x_lds[4] = u_lds(XB_DO * 2 + (w + 3) * 1024);
    x_os[4]  = os_d(w + 3);
    x_lds[5] = u_lds(XB_DO * 2 + j5 * 1024);
    x_os[5]  = os_d(j5);
    x_lds[6] = u_lds((w % 2 == 0 ? XB_LSE : XB_D) * 2);
    x_os[6]  = lane * 4;

    // Fragments of one K step or N tile, at a per-lane LDS address plus an immediate.
    auto mma_k1    = make_mma(number<MR>{}, number<NT>{}, number<1>{});
    auto lay_b     = [&](auto ld, const auto& c) { return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c); };
    const auto c_0 = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    const auto of_q  = opus::layout_to_offsets<8>(lay_b(number<LD_Q>{}, c_0));
    const auto of_do = opus::layout_to_offsets<8>(lay_b(number<LD_DO>{}, c_0));
    const int ob_q   = (opus::layout_to_offsets<8>(lay_b(number<LD_Q>{}, c_ab))[0] - of_q[0]) * 2;
    const int ob_do  = (opus::layout_to_offsets<8>(lay_b(number<LD_DO>{}, c_ab))[0] - of_do[0]) * 2;
    const int ob_tq  = fmha_bwd_d192_v128::tr_c_order_lane<LD_Q>(lane) * 2;
    const int ob_tdo = fmha_bwd_d192_v128::tr_c_order_lane<LD_DO>(lane) * 2;
    const int ob_ld  = 16 * (lane / 16);  // this lane's float4 run of LSE or D
    int c_tq         = ob_tq - ob_q;      // Q^T's per-lane address is Q's plus this
    auto at   = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto at_f = [](unsigned int a) { return opus::make_smem(reinterpret_cast<float*>(static_cast<__UINTPTR_TYPE__>(a))); };
    // An operand's address in buffer b: the scalar part in SALU, one v_add.
    auto addr = [&](int b, int off, int ob) {
        unsigned int sb = fmha_bwd_d192_v128::lds_addr(smem + b + off);
        unsigned int a = sb + static_cast<unsigned int>(ob);
        return a;
    };
    // The first tile is in buffer 0. The first dK block is of no tile (its dS is 0, but what it
    // reads must be finite): it reads the first tile's Q^T too.
    unsigned int a_q = addr(0, XB_Q, ob_q), a_do = addr(0, XB_DO, ob_do), a_tdo = addr(0, XB_DO, ob_tdo);
    unsigned int a_ld = addr(0, XB_LSE, ob_ld), a_tq = addr(0, XB_Q, ob_tq);
    auto rd_q  = [&](int tn, int ks) __attribute__((always_inline)) -> bvec<8> { return at(a_q).template load<8>(of_q[tn] + ks * 32); };
    auto rd_do = [&](int tn, int ks) __attribute__((always_inline)) -> bvec<8> { return at(a_do).template load<8>(of_do[tn] + ks * 32); };
    // One transposed read of N tile gn (16 columns) of Q or dO, half hh of the tile's 32 queries.
    auto rd_tq = [&](int gn, int hh) __attribute__((always_inline)) { return fmha_bwd_d192_v128::tr_read(at(a_tq), gn * 16 + hh * 16 * LD_Q); };
    auto rd_to = [&](int gn, int hh) __attribute__((always_inline)) { return fmha_bwd_d192_v128::tr_read(at(a_tdo), gn * 16 + hh * 16 * LD_DO); };
    using u32x2_t = fmha_bwd_d192_v128::u32x2_t;
    using u32x4_t = fmha_bwd_d192_v128::u32x4_t;
    auto b_tr = [](const u32x2_t (&hv)[2]) __attribute__((always_inline)) { return u32x4_t{hv[0][0], hv[0][1], hv[1][0], hv[1][1]}; };
    auto barrier = [&] {
        fence();
        __builtin_amdgcn_s_barrier();
        fence();
    };
    constexpr int KSV = D_V / 32, NG = NV + NR;
    static_assert(MR == 3 && NT == 2 && KS == 6 && KSV == 4 && NV == 8 && NG == 12 && NS == 24, "the schedule's shape");

    bvec<8> bq[3], bdo[3];  // B fragments of S^T and dP^T: the one in use and the next two
    u32x2_t tb[3][2];       // B of dK and dV: this N tile's two reads and the next two's
    opus::vector_t<float, NS> v_sp, v_dp;
    opus::vector_t<bf16_t, MR * 8> a_p, a_ds;
    f32x4_t lse4[NT], d4[NT];  // LSE log2 e and -D of the tile's queries
    f32x4_t dn4[NT];           // D of the next tile's, from its read to its -D
#pragma unroll
    for(int e = 0; e < NS; e++)
    {
        v_sp[e] = 0.0f;  // P and dS of the tile before the first are 0
        v_dp[e] = 0.0f;
    }
    // The VGPR file: S^T's C tile tl at v[V_SP + 4 tl], dP^T's at v[V_DP + 4 tl], at the top of the 240.
    constexpr int V_DP = 240 - NS, V_SP = V_DP - NS;
    // Below them: dS's and P's bf16 (the A of dK, of dV), LSE log2 e, -D, the LDS addresses (a_q, a_do,
    // a_tdo, a_ld, a_tq).
    constexpr int V_ADS = V_SP - 4 * MR, V_AP = V_ADS - 4 * MR, V_LSE = V_AP - 4 * NT, V_ND = V_LSE - 4 * NT, V_ADR = V_ND - 5;
    // One MFMA of S^T or dP^T onto C tile tl, its result in the tile's registers (reg): from 0, or
    // from c0 (step 0), or in place.
    auto mmp = [&](auto reg, const bvec<8>& fa, const bvec<8>& fb, auto& vacc, int tl, int step, const f32x4_t* c0 = nullptr) __attribute__((always_inline)) {
        f32x4_t cc = c0 ? *c0 : f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        if(step != 0)
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                cc[p_] = vacc[tl * 4 + p_];
        cc = fmha_bwd_d192_v128::pin_vgpr<decltype(reg)::value>(mma1(fa, fb, cc));
#pragma unroll
        for(int p_ = 0; p_ < 4; p_++)
            vacc[tl * 4 + p_] = cc[p_];
    };
    // The loop's VALU is C++ with its results pinned. The compiler counts the wait states between an
    // MFMA and a VALU read of its result in instructions, not cycles: the schedule keeps those reads
    // far enough behind the MFMA that it adds no s_nop.
    auto pack = [&](const opus::vector_t<float, NS>& v, opus::vector_t<bf16_t, MR * 8>& o, int j) __attribute__((always_inline)) {
        opus::vector_t<bf16_t, 2> h2;
        h2 = __builtin_convertvector((opus::vector_t<float, 2>{v[2 * j], v[2 * j + 1]}), opus::vector_t<bf16_t, 2>);
        o[2 * j]     = h2[0];
        o[2 * j + 1] = h2[1];
    };
    auto lse1 = [&](int x) __attribute__((always_inline)) {  // LSE log2 e (0x3fb8aa3b)
        float l = lse4[x / 4][x % 4];
        l *= LOG2_E;
        lse4[x / 4][x % 4] = l;
    };
    // -D of query index x, in place: D's sign bit.
    auto neg1 = [&](int x) __attribute__((always_inline)) {
        float nd = d4[x / 4][x % 4];
        nd = -nd;
        d4[x / 4][x % 4] = nd;
    };

    // The bounds are in.
    const int q0 = static_cast<int>(bnd_q), k0 = static_cast<int>(bnd_k);
    const int seqlen_q = static_cast<int>(bnd_q >> 32) - q0, seqlen_k = static_cast<int>(bnd_k >> 32) - k0;
    const int kv0      = j * BK;
    // A block past the sequence's keys has nothing to do. (No early return: the two bodies are
    // then a plain if / else, with nothing of one live across the other.)
    if(kv0 < seqlen_k)
    {
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. The block's keys see
    // (part of) query tiles i0 .. T - 1; with no query at all the loop runs one tile of zeros.
    const int causal_offset = seqlen_k - seqlen_q;
    const int T             = (seqlen_q + BQ - 1) / BQ;
    int i0                  = 0;
    if(k.causal && kv0 - causal_offset > 0)
        i0 = (kv0 - causal_offset) / BQ;
    // This wave's 48 keys; rows past seqlen_k read as 0.
    const int row_w  = kv0 + w * WR;
    const int rows_w = seqlen_k - row_w < WR ? (seqlen_k - row_w > 0 ? seqlen_k - row_w : 0) : WR;
    const int c_kv   = row_w + lane % 16;

    // The tiles from if0 on are those this wave's keys all see under the causal mask; the ones
    // before take a mask. Nothing else does: rows past seqlen_k and past seqlen_q read as 0 (K, V;
    // Q, dO, LSE, D), and what they make stays in dK / dV rows that are not stored (keys) or is 0
    // (queries: dO and D - dP^T).
    int if0 = 0;
    if(k.causal && rows_w > 0)
    {
        const int need = row_w + rows_w - 1 - causal_offset;
        if0 = need > 0 ? (need + BQ - 1) / BQ : 0;
    }

    const bf16_t* q_ptr  = q_h + static_cast<long long>(q0) * k.stride_q_n;
    const bf16_t* do_ptr = do_h + static_cast<long long>(q0) * k.stride_do_n;
    // readfirstlane: as above (the lanes' offsets use the same products).
    const int x_rb[4] = {__builtin_amdgcn_readfirstlane(k.stride_q_n * 2),
                         __builtin_amdgcn_readfirstlane((r3q ? k.stride_q_n : k.stride_do_n) * 2),
                         __builtin_amdgcn_readfirstlane(k.stride_do_n * 2), 4};
    const char* x_src[4] = {reinterpret_cast<const char*>(q_ptr),
                            r3q ? reinterpret_cast<const char*>(q_ptr) : reinterpret_cast<const char*>(do_ptr),
                            reinterpret_cast<const char*>(do_ptr),
                            reinterpret_cast<const char*>(ld_h + q0)};
    int x_kb[4];
    fmha_bwd_d192_v128::i32x4_t x_rs[4];
#pragma unroll
    for(int d = 0; d < 4; d++)
    {
        x_kb[d] = BQ * x_rb[d];
        x_rs[d] = fmha_bwd_d192_v128::dma_rsrc(x_src[d] + static_cast<long long>(i0) * BQ * x_rb[d], 0);
    }
    auto x_sizes = [&](int t) {
        int rows = seqlen_q - t * BQ;
        rows = rows < BQ ? rows : BQ;
        rows = __builtin_amdgcn_readfirstlane(rows > 0 ? rows : 0);  // the clamp is a VALU med3: one scalar for the products
#pragma unroll
        for(int d = 0; d < 4; d++)
        {
            int nb;
            nb = rows * x_rb[d];
            x_rs[d] = fmha_bwd_d192_v128::i32x4_t{x_rs[d][0], x_rs[d][1], nb, opus::buffer_default_config()};
        }
    };
    auto x_adv = [&](int d) __attribute__((always_inline)) {
        auto& r = x_rs[d];
        unsigned long long a =
            static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
        a += static_cast<unsigned long long>(static_cast<unsigned int>(x_kb[d]));
        r = fmha_bwd_d192_v128::i32x4_t{static_cast<int>(a), static_cast<int>(a >> 32), r[2], r[3]};
    };
    auto x_go = [&](int xr, unsigned int lb) __attribute__((always_inline)) {
        const int d = xr < 3 ? 0 : (xr == 3 ? 1 : (xr < XR - 1 ? 2 : 3));
        if(xr == XR - 1)
            fmha_bwd_d192_v128::dma_round<4>(x_rs[d], lb + x_lds[xr], x_os[xr]);
        else
            fmha_bwd_d192_v128::dma_round<16>(x_rs[d], lb + x_lds[xr], x_os[xr]);
    };

    // The first two tiles to LDS (buffers 0 and 1).
    {
        unsigned int lb0 = lds_of(0), lb1 = lds_of(XB_SZ);
        x_sizes(i0);
#pragma unroll
        for(int xr = 0; xr < XR; xr++)
            x_go(xr, lb0);
#pragma unroll
        for(int d = 0; d < 4; d++)
            x_adv(d);
        x_sizes(i0 + 1);
#pragma unroll
        for(int xr = 0; xr < XR; xr++)
            x_go(xr, lb1);
        // Descriptor 0 goes a tile on past its rounds, the others in front of theirs.
        x_adv(0);
        x_sizes(i0 + 2);
    }
    // This wave's keys as the A operand of S^T and dP^T, in registers.
    auto g_kn = opus::make_gmem(kn_h + static_cast<long long>(k0 + row_w) * k.stride_k_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
    auto g_kr = opus::make_gmem(kn_h + D_NOPE + static_cast<long long>(k0 + row_w) * k.stride_k_n,
                                rec(static_cast<long long>(rows_w) * k.stride_k_n, 2));
    auto g_v  = opus::make_gmem(v_h + static_cast<long long>(k0 + row_w) * k.stride_v_n,
                                rec(static_cast<long long>(rows_w) * k.stride_v_n, 2));
    const auto v_kn = g_kn.template load<8>(opus::partition_layout_a<8>(mma_kn, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
    const auto v_kr = g_kr.template load<8>(opus::partition_layout_a<8>(mma_kr, opus::make_tuple(k.stride_k_n, 1_I), c_ab));
    const auto v_vw = g_v.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_v_n, 1_I), c_ab));
    // A of S^T is [M tile][K step][8]; nope and rope interleave per M tile.
    typename decltype(mma_s)::vtype_a v_kw;
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int ks = 0; ks < KS; ks++)
#pragma unroll
            for(int e = 0; e < 8; e++)
            {
                if(ks < KS_N)
                    v_kw[(m * KS + ks) * 8 + e] = v_kn[(m * KS_N + ks) * 8 + e];
                else
                    v_kw[(m * KS + ks) * 8 + e] = v_kr[(m * KS_R + ks - KS_N) * 8 + e];
            }

    // dV's accumulators, zeroed behind the loads of K and V (dK's come zeroed from the kernel).
    f32x4_t acc_dv[MR][NV];
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int n = 0; n < NV; n++)
        {
            acc_dv[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        }
    opus::static_for<MR * NV>([&](auto I) {
        constexpr int i = decltype(I)::value;
        acc_dv[i / NV][i % NV] = fmha_bwd_d192_v128::pin_agpr<A_DV + 4 * i>(acc_dv[i / NV][i % NV]);
    });
    {
        // Loop-carried scalars, at the top of the iteration of S^T tile ts: lbc, lbp the LDS
        // address of tile ts - 1's buffer and of tile ts - 2's (the DMA's target, tile ts + 2
        // goes there); dlt the LDS addresses' step from tile ts - 1's buffer to tile ts's (from
        // tile ts's to the next past the scalar C_DLT); rc tile ts's buffer in the ring (0 .. 3), dn
        // the step from it to the next (-3 buffers past the last, without flags: single scalars
        // go anywhere); ra = seqlen_q - (ts + 4) BQ (negative: tile ts + 3 is not a full one),
        // fd = ts - if0 (negative: mask), z = ts - T.
        constexpr int SZB = XB_SZ * 2;
        static_assert(XNB == 4, "the ring's wrap below");
        int rc = 0, dn = 0;
        int ring = -XNB * SZB, step = SZB;
        int dlt = -(XNB - 1) * SZB;
        unsigned int lbc = lds_of((XNB - 1) * XB_SZ), lbp = lds_of((XNB - 2) * XB_SZ);
        int ra = seqlen_q - (i0 + 4) * BQ, fd = i0 - if0, z = i0 - T;

        namespace sch = fmha_bwd_d192_v128_dkdv_sched;
        using sch::reads;
        using sch::reads_head;
        // The loop's fragments in order: S^T's, dP^T's, dK's, dV's. sch's rules give every element its three steps
        // and every pair its two packs, the loop its scalars and addresses.
        constexpr int NF_S = NT * KS, NF_P = NT * KSV, I_P = NF_S, I_K = I_P + NF_P, I_V = I_K + NG, NL = I_V + NV;
        static_assert(sch::MR == MR && sch::NF_S == NF_S && sch::NF_P == NF_P && sch::NF_K == NG && sch::NF_V == NV && sch::NL == NL &&
                          sch::has_dma(XR - 1) && !sch::has_dma(XR),
                      "the loop of the rules");
        static_assert(sch::total(sch::scalars) == sch::C_N && sch::total(sch::pds) == NS / 2 && sch::total(sch::lses) == 4 * NT &&
                          sch::total(sch::fmas) == NS && sch::total(sch::exps) == NS && sch::total(sch::muls) == NS &&
                          sch::total(sch::pps) == NS / 2 && sch::total(sch::negs) == 4 * NT && sch::total(sch::addrs) == 5,
                      "the loop's work");
        constexpr unsigned int S_EXPD = sch::exp_done(sch::exps(I_P).first);

        // The GEMMs, a fragment at a time: its MR MFMAs, one per M tile, on its B operand in the ring; behind(F) is
        // what rides behind them, as the per_step of the MLA kernels' compute_qk; a fence ends the fragment.
        // S^T of tile ts: fragment f is N tile f / 6, K step f % 6.
        auto gemm_s = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NF_S>([&](auto F) {
                constexpr int f = decltype(F)::value, tn = f / KS, ks = f % KS;
                opus::static_for<MR>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    bvec<8> fa;
#pragma unroll
                    for(int x = 0; x < 8; x++)
                        fa[x] = v_kw[(j * KS + ks) * 8 + x];
                    mmp(number<V_SP + 4 * (j * NT + tn)>{}, fa, bq[f % 3], v_sp, j * NT + tn, ks == 0 ? 0 : 1);
                });
                behind(F);
                fence();
            });
        };
        // dP^T of tile ts: fragment f is N tile f / 4, K step f % 4; the accumulators start at -D.
        auto gemm_dp = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NF_P>([&](auto F) {
                constexpr int f = decltype(F)::value, tn = f / KSV, ks = f % KSV;
                opus::static_for<MR>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    bvec<8> fa;
#pragma unroll
                    for(int x = 0; x < 8; x++)
                        fa[x] = v_vw[(j * KSV + ks) * 8 + x];
                    mmp(number<V_DP + 4 * (j * NT + tn)>{}, fa, bdo[f % 3], v_dp, j * NT + tn, ks == 0 ? 0 : 1,
                        ks == 0 ? &d4[tn] : nullptr);
                });
                behind(F);
                fence();
            });
        };
        // dK of the tile before (of the last tile in the tail): a fragment is an N tile of Q^T, 16 of dK's columns.
        auto gemm_dk = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NG>([&](auto G) {
                constexpr int gn = decltype(G)::value;
                opus::static_for<MR>([&](auto M) {
                    constexpr int m = decltype(M)::value;
                    auto& ac = [&]() -> f32x4_t& {
                        if constexpr(gn < NV)
                            return acc_dkn[m][gn];
                        else
                            return acc_dkr[m][gn - NV];
                    }();
                    ac = fmha_bwd_d192_v128::pin_agpr<A_DK + 4 * (m * NG + gn)>(
                        mma1(__builtin_bit_cast(bvec<8>, fmha_bwd_d192_v128::frag8(a_ds, m)),
                             __builtin_bit_cast(bvec<8>, b_tr(tb[gn % 3])), ac));
                });
                behind(G);
                fence();
            });
        };
        // dV of tile ts: a fragment is an N tile of dO^T, 16 of dV's columns.
        auto gemm_dv = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NV>([&](auto G) {
                constexpr int gn = decltype(G)::value;
                opus::static_for<MR>([&](auto M) {
                    constexpr int m = decltype(M)::value;
                    acc_dv[m][gn] = fmha_bwd_d192_v128::pin_agpr<A_DV + 4 * (m * NV + gn)>(
                        mma1(__builtin_bit_cast(bvec<8>, fmha_bwd_d192_v128::frag8(a_p, m)),
                             __builtin_bit_cast(bvec<8>, b_tr(tb[gn % 3])), acc_dv[m][gn]));
                });
                behind(G);
                fence();
            });
        };

        // Elements 2 j, 2 j + 1 of v to bf16, into o's at register R.
        auto pack_p = [&](auto R, const opus::vector_t<float, NS>& v, opus::vector_t<bf16_t, MR * 8>& o, int j) __attribute__((always_inline)) {
            auto h2 = __builtin_convertvector((opus::vector_t<float, 2>{v[2 * j], v[2 * j + 1]}), opus::vector_t<bf16_t, 2>);
            h2 = __builtin_bit_cast(opus::vector_t<bf16_t, 2>,
                                    fmha_bwd_d192_v128::pin_vgpr<decltype(R)::value>(__builtin_bit_cast(opus::u32_t, h2)));
            o[2 * j]     = h2[0];
            o[2 * j + 1] = h2[1];
        };

        // What rides behind the loop's fragment i: a chunk for each kind of work. A chunk takes the kind's items
        // first .. first + n - 1, as sch's rule of the kind gives them to the fragment; each item's result goes to
        // its own register.
        // dS of the tile before to bf16, pair by pair: the A operand of the dK block.
        auto pd_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::pds(decltype(I)::value).first, n = sch::pds(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int j = first + decltype(X)::value;
                    pack_p(number<V_ADS + j>{}, v_dp, a_ds, j);
                });
        };
        // LSE log2 e of the tile's queries, in place.
        auto lse_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::lses(decltype(I)::value).first, n = sch::lses(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int x = first + decltype(X)::value;
                    lse4[x / 4][x % 4] = fmha_bwd_d192_v128::pin_vgpr<V_LSE + x>(lse4[x / 4][x % 4] * LOG2_E);
                });
        };
        // The softmax, in place: S^T scale - LSE, the elements N tile by N tile.
        auto scale_sub_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::fmas(decltype(I)::value).first, n = sch::fmas(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem(first + decltype(X)::value);
                    v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(__builtin_fmaf(v_sp[e], sc2, -lse4[e / 4 % NT][e % 4]));
                });
        };
        // Its exp, P.
        auto exp_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::exps(decltype(I)::value).first, n = sch::exps(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem(first + decltype(X)::value);
                    v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(__builtin_amdgcn_exp2f(v_sp[e]));
                });
        };
        // dS = P (dP^T - D) in fp32, in dP^T's place.
        auto ds_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::muls(decltype(I)::value).first, n = sch::muls(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem(first + decltype(X)::value);
                    v_dp[e] = fmha_bwd_d192_v128::pin_vgpr<V_DP + e>(v_dp[e] * v_sp[e]);
                });
        };
        // P to bf16, the pairs N tile by N tile: the A operand of the dV block.
        auto pp_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::pps(decltype(I)::value).first, n = sch::pps(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int j = sch::elem(2 * (first + decltype(X)::value)) / 2;
                    pack_p(number<V_AP + j>{}, v_sp, a_p, j);
                });
        };
        // -D of the next tile's queries, from the D read behind dP^T's fragments 1 and 2.
        auto neg_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::negs(decltype(I)::value).first, n = sch::negs(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int x = first + decltype(X)::value;
                    d4[x / 4][x % 4] = fmha_bwd_d192_v128::pin_vgpr<V_ND + x>(-dn4[x / 4][x % 4]);
                });
        };
        // The LDS addresses a tile on, in their order: LSE's and D's, Q^T's from Q's, then Q's, dO's, dO^T's.
        auto addr_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = sch::addrs(decltype(I)::value).first, n = sch::addrs(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int a = first + decltype(X)::value;
                    if constexpr(a == 0)
                        a_ld = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 3>(a_ld + static_cast<unsigned int>(dlt));
                    else if constexpr(a == 1)
                        a_tq = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 4>(a_q + static_cast<unsigned int>(c_tq));
                    else if constexpr(a == 2)
                        a_q = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 0>(a_q + static_cast<unsigned int>(dlt));
                    else if constexpr(a == 3)
                        a_do = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 1>(a_do + static_cast<unsigned int>(dlt));
                    else
                        a_tdo = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 2>(a_tdo + static_cast<unsigned int>(dlt));
                });
        };
        // The chunks of the loop's fragment i, in the order they issue: the VALU fill the MFMAs' shares in the code's
        // order, the exps come last.
        auto softmax_chunk = [&](auto I) __attribute__((always_inline)) {
            pd_chunk(I);
            lse_chunk(I);
            scale_sub_chunk(I);
            ds_chunk(I);
            pp_chunk(I);
            neg_chunk(I);
            addr_chunk(I);
            exp_chunk(I);
        };
        // The schedule of a fragment, one sched_mfma per MFMA. With STEP its first MFMA takes a DMA descriptor's
        // step alone, with HEAD an LDS read and a VALU; with DMA its second a DMA round. The last takes the wait,
        // the READS LDS reads and what VALU fits beside them. Every other MFMA has the same share: two VALU and an
        // exp, of which a fragment gives it the one or the other. Of the fragment's SC scalars one goes behind each
        // of these, from the first, and if there are more behind the last, in place of a VALU.
        auto sched_chunk = [&](auto SC, auto STEP, auto DMA, auto READS, auto HEAD) __attribute__((always_inline)) {
            constexpr int n_sc = decltype(SC)::value, n_step = decltype(STEP)::value, n_dma = decltype(DMA)::value;
            constexpr int n_rd = decltype(READS)::value, n_head = decltype(HEAD)::value;
            constexpr int s_last = n_sc > MR - 1 - n_step - n_dma ? 1 : 0;
            static_assert(n_sc <= MR - n_step - n_dma && n_rd + s_last <= 2 && n_step + n_head <= 1, "a fragment's MFMAs");
            opus::static_for<MR>([&](auto J) {
                constexpr int j = decltype(J)::value;
                // A scalar is two instructions for the schedule: the op and its anchor.
                constexpr int s2 = 2 * (j - n_step - (n_dma && j > 1) < n_sc);
                if constexpr(n_step && j == 0)
                    fmha_bwd_d192_v128::sched_mfma<2, 0>();
                else if constexpr(n_head && j == 0)
                    fmha_bwd_d192_v128::sched_mfma<s2, 1, 0, 1>();
                else if constexpr(n_dma && j == 1)
                    fmha_bwd_d192_v128::sched_mfma<1, 0, 0, 0, 1>();
                else if constexpr(j == MR - 1)
                    fmha_bwd_d192_v128::sched_mfma<1 + 2 * s_last, 2 - n_rd - s_last, 0, n_rd>();
                else
                    fmha_bwd_d192_v128::sched_mfma<s2, 2, 1>();
            });
        };

        // S^T of tile ts. Behind fragment f: 0 .. 6 a DMA round of tile ts + 2 (0 .. 3 a descriptor's step too);
        // 0, 1 LSE's reads; 2 .. 5 LSE log2 e of N tile 0, one each; 4 .. 8 dS of the tile before to bf16; 4 .. 11
        // the loop's scalars; 6 .. 10 S^T scale - LSE of N tile 0, in what the others leave; 10 two LSE log2 e of
        // N tile 1; 11 the first two exps. Returns a negative number if the sizes of tile ts + 3 are to be set or
        // S^T is to be masked (rare_fix).
        auto s_block = [&]() __attribute__((always_inline)) {
            int rare = 0;
            // The loop's scalar c, anchored where it stands.
            auto scalar = [&](auto C) __attribute__((always_inline)) {
                constexpr int c = decltype(C)::value;
                // The ring a buffer on: rc = (rc + 1) & 3, dn = rc == 0 ? -3 buffers : 1 buffer.
                if constexpr(c == sch::C_RC_INC)
                    { rc += 1; here(rc); }
                else if constexpr(c == sch::C_RC_AND)
                    { rc &= 3; here(rc); }
                else if constexpr(c == sch::C_DN_SUB)
                    { dn = rc - 1; here(dn); }
                else if constexpr(c == sch::C_DN_SHR)
                    { dn >>= 31; here(dn); }
                else if constexpr(c == sch::C_DN_AND)
                    { dn &= ring; here(dn); }
                // Past the last DMA round: tile ts's buffer, the DMA's next target (the buffer before it: no
                // copy, which would be no instruction of the schedule's), then the address step a tile on.
                else if constexpr(c == sch::C_LBC)
                    { lbc += static_cast<unsigned int>(dlt); here(lbc); }
                else if constexpr(c == sch::C_LBP)
                    { lbp = lbc - static_cast<unsigned int>(dlt); here(lbp); }
                else if constexpr(c == sch::C_DLT)
                    { dlt = dn + step; here(dlt); }
                // The rare cases' sign from the S^T tile's ra and fd, then the counters on.
                else if constexpr(c == sch::C_RARE)
                    { rare = ra | fd; here(rare); }
                else if constexpr(c == sch::C_Z)
                    { z += 1; here(z); }
                else if constexpr(c == sch::C_RA)
                    { ra -= BQ; here(ra); }
                else if constexpr(c == sch::C_FD)
                    { fd += 1; here(fd); }
            };
            using sch::has_dma;
            using sch::has_step;
            // Behind the first MFMA: a DMA descriptor a tile on (1 .. 3 in front of their rounds, 0 past its), or
            // the fragment's first scalar.
            auto head_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                if constexpr(has_step(f))
                    x_adv(sch::S_ADV[f]);
                else if constexpr(sch::scalars(f).n > 0)
                    scalar(number<sch::scalars(f).first>{});
            };
            // Behind the second MFMA: DMA round f of tile ts + 2.
            auto dma_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                if constexpr(has_dma(f))
                    x_go(f, lbp);
            };
            // The fragment's other scalars: one behind each of the other MFMAs, from the first.
            auto scalar_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value, lead = !has_step(f) && sch::scalars(f).n > 0 ? 1 : 0;
                constexpr int first = sch::scalars(f).first + lead, n = sch::scalars(f).n - lead;
                if constexpr(n > 0)
                    opus::static_for<n>([&](auto X) { scalar(number<first + decltype(X)::value>{}); });
            };
            // Behind the last MFMA: the wait for the next fragment's Q, in the place that is free only for a scalar
            // (in flight are the reads behind the fragment before), then the ring's refill; behind the last
            // fragments, dO's first three for the dP^T block. Behind the first two, LSE's N tiles.
            auto refill_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                opus::s_waitcnt_lgkmcnt(number<reads(f - 1)>{});
                if constexpr(f + 3 < NF_S)
                    bq[f % 3] = rd_q((f + 3) / KS, (f + 3) % KS);
                else if constexpr(f == NF_S - 3)
                    bdo[0] = rd_do(0, 0);
                else if constexpr(f == NF_S - 2)
                {
                    bdo[1] = rd_do(0, 1);
                    bdo[2] = rd_do(0, 2);
                }
                if constexpr(f < NT)
                    lse4[f] = at_f(a_ld).template load<4>(16 * f);
            };
            gemm_s([&](auto F) {
                constexpr int f = decltype(F)::value;
                head_chunk(F);
                dma_chunk(F);
                scalar_chunk(F);
                softmax_chunk(F);
                refill_chunk(F);
                sched_chunk(number<sch::scalars(f).n>{}, number<has_step(f)>{}, number<has_dma(f)>{}, number<reads(f)>{}, number<0>{});
            });
            return rare;
        };
        // The rare cases: tile ts + 3 is not a full one (its sizes), this wave's keys do not all
        // see tile ts under the causal mask: the elements masked take P = 0 where the exp is made
        // already, else an S^T (or S^T scale - LSE) that exps to 0.
        auto rare_fix = [&](int ts) {
            if(seqlen_q - (ts + 4) * BQ < 0)
                x_sizes(ts + 3);
            if(ts < if0)
            {
                // Element e is key row_w + 16 (e / 8) + lane % 16 and query BQ ts + 16 (e / 4 % 2)
                // + 4 (lane / 16) + e % 4, seen if query + causal_offset >= key: xm[M tile] is
                // the first query of the tile the lane's key sees, less the lane's 4 (lane / 16).
                int xm[MR];
#pragma unroll
                for(int m = 0; m < MR; m++)
                    xm[m] = c_kv + 16 * m - causal_offset - ts * BQ - c_q;
                // The mask, four elements (one C tile) at a time: four compares into scalar pairs,
                // then the four selects, so that each select stands three instructions behind its
                // compare.
                opus::static_for<NS / 4>([&](auto G) {
                    constexpr int gt = decltype(G)::value, c0 = 16 * (gt % NT), e0 = 4 * gt;
                    // The four compares, then the four selects: each select stands three behind its compare.
                    bool in[4];
#pragma unroll
                    for(int p_ = 0; p_ < 4; p_++)
                        in[p_] = xm[gt / NT] <= c0 + p_;
                    __builtin_amdgcn_sched_barrier(0);
                    opus::static_for<4>([&](auto P) {
                        constexpr int e = e0 + decltype(P)::value;
                        const float m = (S_EXPD >> e & 1) ? 0.0f : -1e30f;
                        v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(in[decltype(P)::value] ? v_sp[e] : m);
                    });
                });
            }
        };
        // dP^T of tile ts. Behind fragment f: 0 the other two LSE log2 e and LSE's address a tile on; 0 .. 2 and 4
        // S^T scale - LSE of N tile 1; 3, 6, 7 exps; 4, 5 the first four dS (dP^T's N tile 0 is done); 5 the first
        // two -D of the next tile. Behind the last MFMA: the wait for the next fragment's dO, then the dO ring's
        // refill (behind the next fragment's first MFMA for fragments 0 and 4, which start from -D); behind the
        // last three fragments, Q^T's first three N tiles for the dK block. Behind fragments 1 and 2, D of the
        // next tile's N tiles 0 and 1.
        auto dp_block = [&]() __attribute__((always_inline)) {
            gemm_dp([&](auto F) {
                constexpr int f = decltype(F)::value, i = I_P + f, nd = f == 1 || f == 2 ? f - 1 : -1;
                if constexpr(reads_head(i))
                    bdo[(f - 1) % 3] = rd_do((f + 2) / KSV, (f + 2) % KSV);
                softmax_chunk(number<i>{});
                opus::s_waitcnt_lgkmcnt(number<reads(i - 1) + reads_head(i)>{});
                if constexpr(f + 3 >= NF_P)
                {
                    tb[f + 3 - NF_P][0] = rd_tq(f + 3 - NF_P, 0);
                    tb[f + 3 - NF_P][1] = rd_tq(f + 3 - NF_P, 1);
                }
                else if constexpr(f % KSV != 0)
                    bdo[f % 3] = rd_do((f + 3) / KSV, (f + 3) % KSV);
                if constexpr(nd >= 0)
                    dn4[nd] = at_f(a_ld).template load<4>(64 + 16 * nd);
                sched_chunk(number<0>{}, number<0>{}, number<0>{}, number<reads(i)>{}, number<reads_head(i)>{});
            });
        };
        // dK of the tile before; TAIL: of the last tile, behind the loop. In the loop, behind fragment f: 0 .. 7 the
        // other exps; 8 .. 10 P to bf16 for the dV block; 11 dS. Behind the last MFMA, the wait for the next N tile
        // of Q^T and the Q^T ring's refill; the last three read dO^T's first three N tiles for the dV block. The
        // tail has the waits and the refills alone.
        auto dk_block = [&](auto TAIL) __attribute__((always_inline)) {
            constexpr bool tail = decltype(TAIL)::value;
            gemm_dk([&](auto G) {
                constexpr int gn = decltype(G)::value, i = I_K + gn;
                // The reads behind this fragment and behind the one before (the tail's ring ends with Q^T).
                constexpr int n_rd = gn + 3 < NG ? 2 : (tail ? 0 : reads(i));
                constexpr int n_before = gn == 0 || gn + 2 < NG ? 2 : (tail ? 0 : reads(i - 1));
                if constexpr(!tail)
                    softmax_chunk(number<i>{});
                opus::s_waitcnt_lgkmcnt(number<n_before>{});
                if constexpr(gn + 3 < NG)
                {
                    tb[gn % 3][0] = rd_tq(gn + 3, 0);
                    tb[gn % 3][1] = rd_tq(gn + 3, 1);
                }
                else if constexpr(!tail)
                {
                    tb[gn % 3][0] = rd_to(gn + 3 - NG, 0);
                    tb[gn % 3][1] = rd_to(gn + 3 - NG, 1);
                }
                sched_chunk(number<0>{}, number<0>{}, number<0>{}, number<n_rd>{}, number<0>{});
            });
        };
        // dV of tile ts. Behind fragment f: 0 .. 3 the last of dS; 4, 5 the other -D of the next tile; 5, 6 the
        // other addresses a tile on. Behind the last MFMA, the wait for the next N tile of dO^T and the dO^T
        // ring's refill; the last three read the next tile's first three fragments of Q for its S^T block, and the
        // last waits for this wave's DMA loads too (issued in this iteration's S^T block: in before the barrier).
        auto dv_block = [&]() __attribute__((always_inline)) {
            gemm_dv([&](auto G) {
                constexpr int gn = decltype(G)::value, i = I_V + gn;
                softmax_chunk(number<i>{});
                if constexpr(gn == NV - 1)
                    fmha_bwd_d192_v128::wait_loads_reads<reads(i - 1)>();
                else
                    opus::s_waitcnt_lgkmcnt(number<reads(i - 1)>{});
                if constexpr(gn + 3 < NV)
                {
                    tb[gn % 3][0] = rd_to(gn + 3, 0);
                    tb[gn % 3][1] = rd_to(gn + 3, 1);
                }
                else if constexpr(gn == NV - 3)
                    bq[0] = rd_q(0, 0);
                else if constexpr(gn == NV - 2)
                {
                    bq[1] = rd_q(0, 1);
                    bq[2] = rd_q(0, 2);
                }
                sched_chunk(number<0>{}, number<0>{}, number<0>{}, number<reads(i)>{}, number<0>{});
            });
        };

        // K and V are in registers and this wave's loads of the first two tiles in LDS (wait_loads: the
        // compiler then waits for none of them inside the loop); every wave's are behind the barrier.
        __builtin_amdgcn_sched_barrier(0);  // dV's zeros in front of the wait (writes of a pinned register go to the end of their region)
        fmha_bwd_d192_v128::wait_loads();
        barrier();
        bq[0] = rd_q(0, 0);
        bq[1] = rd_q(0, 1);
        bq[2] = rd_q(0, 2);
        lds_wait(number<2>{});
        // -D of the first tile's queries, where its dP^T's accumulators start; the loop reads each
        // next tile's D in its dP^T block.
#pragma unroll
        for(int tn = 0; tn < NT; tn++)
            d4[tn] = at_f(a_ld).template load<4>(64 + 16 * tn);
#pragma unroll
        for(int x = 0; x < 4 * NT; x++)
            neg1(x);
        // The loop's code starts on a 64-byte boundary.
        do
        {
            // Every wave is past dK's reads of tile ts - 2, whose buffer this S^T block's DMA
            // overwrites, and has its loads of tile ts + 1 in.
            barrier();
            const int rare = s_block();
            if(__builtin_expect(rare < 0, 0))
                rare_fix(z - 1 + T);
            dp_block();
            dk_block(number<0>{});
            dv_block();
        } while(__builtin_expect(z < 0, 1));
        // The last tile's dK: its dS to bf16, Q^T's first three N tiles (the ring's head, which in the loop the
        // dP^T block reads; Q^T's address the last iteration left on the last tile). The first N tile is in.
#pragma unroll
        for(int jp = 0; jp < NS / 2; jp++)
            pack(v_dp, a_ds, jp);
#pragma unroll
        for(int x = 0; x < 6; x++)
            tb[x / 2][x % 2] = rd_tq(x / 2, x % 2);
        lds_wait(number<4>{});
        dk_block(number<1>{});
    }
    // The loads still out go to LDS, which the epilogue uses.
    fmha_bwd_d192_v128::wait_loads();

    // dK and dV, a row per key. The C fragments hold 4 columns of 16 rows per instruction, so each
    // wave transposes its 48 rows through LDS first and stores whole rows; rows past seqlen_k fall
    // outside the descriptors and are dropped.
    fmha_bwd_d192_v128::block_sync();
    // A tile's reads where it is used, behind the barrier: each tile re-pinned there, a fence behind it (copies out
    // of a pinned register go to the top of their region).
    __builtin_amdgcn_sched_barrier(0);
    {
        static_assert(NW * WR * LD_O <= SMEM_ELEMS && WR * (D_QK / 8) % 64 == 0 && WR * (D_V / 8) % 64 == 0,
                      "staging fits; whole instructions");
        bf16_t* s_o = smem + w * WR * LD_O;
        // C element (m, n, e): row 16 m + lane % 16, column 16 n + 4 (lane / 16) + e.
        const int c_row = lane % 16, c_col = 4 * (lane / 16);
        const long long r0 = k0 + row_w;
        // dK [48][192]: the nope part's tiles, then the rope part's.
        opus::static_for<MR * (NV + NR)>([&](auto I) {
            constexpr int i = decltype(I)::value, m = i / (NV + NR), n = i % (NV + NR);
            bf16_t* o = s_o + (16 * m + c_row) * LD_O + c_col;
            if constexpr(n < NV)
                *reinterpret_cast<bvec<4>*>(o + 16 * n) =
                    opus::cast<bf16_t>(fmha_bwd_d192_v128::pin_agpr<A_DK + 4 * i>(acc_dkn[m][n]) * k.softmax_scale);
            else
                *reinterpret_cast<bvec<4>*>(o + D_NOPE + 16 * (n - NV)) =
                    opus::cast<bf16_t>(fmha_bwd_d192_v128::pin_agpr<A_DK + 4 * i>(acc_dkr[m][n - NV]) * k.softmax_scale);
            if constexpr(i % 2 == 1)  // two tiles a fence: their two LDS writes one ds_write2
                __builtin_amdgcn_sched_barrier(0);
        });
        auto g_dk = opus::make_gmem(reinterpret_cast<bf16_t*>(k.ptr_dk) + r0 * k.stride_dk_n + static_cast<long long>(h) * k.stride_dk_h,
                                    rec(static_cast<long long>(rows_w) * k.stride_dk_n, 2));
        // 64 lanes x 8 bf16 on end along the rows.
        constexpr int OK = D_QK / 8;
#pragma unroll
        for(int it = 0; it < WR * OK / 64; it++)
        {
            const int ch = it * 64 + lane, row = ch / OK, col = ch % OK * 8;
            g_dk.template store<8>(*reinterpret_cast<const vec8_t*>(s_o + row * LD_O + col), row * k.stride_dk_n + col);
        }
        // dV [48][128], over the same LDS.
        opus::static_for<MR * NV>([&](auto I) {
            constexpr int i = decltype(I)::value, m = i / NV, n = i % NV;
            *reinterpret_cast<bvec<4>*>(s_o + (16 * m + c_row) * LD_O + 16 * n + c_col) =
                opus::cast<bf16_t>(fmha_bwd_d192_v128::pin_agpr<A_DV + 4 * i>(acc_dv[m][n]));
            if constexpr(i % 2 == 1)  // two tiles a fence: their two LDS writes one ds_write2
                __builtin_amdgcn_sched_barrier(0);
        });
        auto g_dv = opus::make_gmem(reinterpret_cast<bf16_t*>(k.ptr_dv) + r0 * k.stride_dv_n + static_cast<long long>(h) * k.stride_dv_h,
                                    rec(static_cast<long long>(rows_w) * k.stride_dv_n, 2));
        constexpr int OV = D_V / 8;
#pragma unroll
        for(int it = 0; it < WR * OV / 64; it++)
        {
            const int ch = it * 64 + lane, row = ch / OV, col = ch % OV * 8;
            g_dv.template store<8>(*reinterpret_cast<const vec8_t*>(s_o + row * LD_O + col), row * k.stride_dv_n + col);
        }
    }
    }
}

// The dQ pass's loop for a wave of MR M tiles: 32 fragments in the loop's order, S's 12 (fragment f: N tile f / 6, K
// step f % 6), dP's 8 (N tile f / 4, K step f % 4) and dQ's 12 (an N tile each), every fragment MR MFMAs, one per M
// tile. The rules below say where the loop's work rides behind them, each kind of work in an order of its own:
//   scalars  the loop's scalars, in the order of C_... (S only)
//   packs    dS of the tile before to bf16, two elements each: the A operand of the dQ block (S only)
//   fmas     S scale - LSE             exps     its exp, P
//   muls     P (dP - D), dS in fp32: dP's accumulators start at -D
//   addrs    an LDS address a tile on: K nope's, K rope's, then K^T's two from V's, then V's
// Element e of a wave's 8 MR S / dP values per lane is C tile (M tile e / 8, N tile e / 4 % 2); packs and fma take
// the elements N tile by N tile (elem), exp and mul the last M tile's behind the others' (elem_exp).
// An MFMA leaves three places behind it, the first free only for a scalar or a wait, the others for a VALU or an
// LDS read each (an exp takes both, a DMA round the three). A fragment's last MFMA takes the wait and the fragment's
// LDS reads; every other one has the same share, two VALU and one exp (the first of S's first four, with a DMA
// descriptor's step, nothing: a VALU behind the step's two scalars costs a cycle; the second of S's first six the DMA
// round). The shares are filled in the code's order, so a fragment is of exps, one an MFMA, or of VALU: with both,
// its first MFMAs would get an exp and VALU, more than their three places. The work keeps its order: an fma behind
// the last MFMA of its C tile in S and a mul behind that in dP by four MFMAs or more (a VALU reads an MFMA's
// result); an exp in a fragment behind its fma's and a mul behind its exp's (a value's register is written at the
// end of the fragment that makes it); an address behind the last LDS read at the old one.
namespace fmha_bwd_d192_v128_dq_sched {
// The loop's scalars, in the order the S block takes them.
enum : signed char { C_RC_INC, C_RC_AND, C_DN_SUB, C_DN_SHR, C_DN_AND, C_LBC, C_LBP, C_DLT, C_RARE, C_Z, C_RA, C_NF, C_N };
// The blocks' fragments and the loop's index of each block's first.
constexpr int NF_S = 12, NF_P = 8, NF_Q = 12, I_P = NF_S, I_Q = I_P + NF_P, NL = I_Q + NF_Q;
// The items first .. first + n - 1 of a kind of work.
struct span
{
    int first, n;
};
constexpr int clamp(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
constexpr int none(int)
{
    return 0;
}
// The i-th of a wave's 8 MR elements N tile by N tile, and with the last M tile's behind the others'.
constexpr int elem(int MR, int i)
{
    return 8 * (i / 4 % MR) + 4 * (i / (4 * MR)) + i % 4;
}
constexpr int elem_exp(int MR, int i)
{
    return i < 8 * (MR - 1) ? elem(MR - 1, i) : i;
}
// Bit e set if element e's exp is among the first n (those of the S block: the mask of a tile not seen in full is
// applied behind it, and those elements take P = 0, the others an S that exps to 0).
constexpr unsigned int exp_done(int MR, int n)
{
    unsigned int m = 0;
    for(int i = 0; i < n; i++)
        m |= 1u << elem_exp(MR, i);
    return m;
}
// The DMA descriptor that goes a tile on behind each of the S block's first four fragments.
inline constexpr signed char S_ADV[4] = {2, 1, 0, 3};

// S's first four fragments step a DMA descriptor behind their first MFMA, its first six issue a DMA round behind
// their second.
constexpr bool has_step(int i)
{
    return i < 4;
}
constexpr bool has_dma(int i)
{
    return i < 6;
}
// The rings of the B operands: a block's fragment f is in entry f % 3 of its ring (bq, bv, tb), and behind
// fragment f's last MFMA that entry is refilled with the fragment three ahead, as the MLA kernels refill their K
// ring behind a step's MFMAs; a block's last three fragments read the next block's first three instead (dQ's the
// next tile's S's). S's and dQ's last fragment, in front of a branch, reads nothing: the one before it reads two.
// reads(i): the LDS reads behind the loop's fragment i (the one before its first is its last); an N tile of K^T is
// two.
constexpr int reads(int i)
{
    i = (i + NL) % NL;
    // S: K's fragment three ahead; its last three read V's first three: one, two, none (in front of the branch).
    if(i < I_P)
    {
        if(i + 3 < NF_S)
            return 1;
        return i == NF_S - 3 ? 1 : (i == NF_S - 2 ? 2 : 0);
    }
    // dP: V's fragment three ahead; its last three read K^T's first three N tiles.
    if(i < I_Q)
        return i - I_P + 3 < NF_P ? 1 : 2;
    // dQ: K^T's N tile three ahead; its last three read the next tile's first three fragments of K: one, two, none
    // (in front of the branch).
    const int g = i - I_Q;
    if(g + 3 < NF_Q)
        return 2;
    return g == NF_Q - 3 ? 1 : (g == NF_Q - 2 ? 2 : 0);
}
// The loop's scalars: one behind each of S's fragments 0 and 1, one behind each of 4, 5 and 6, then two a fragment.
constexpr span scalars(int i)
{
    return i < 2 ? span{i, 1} : i >= 4 && i < 7 ? span{i - 2, 1} : i >= 7 && i < I_P ? span{2 * i - 9, clamp(C_N - (2 * i - 9), 0, 2)} : span{0, 0};
}
// All the loop's items of a kind.
template<class K>
constexpr int total(K kind)
{
    int n = 0;
    for(int i = 0; i < NL; i++)
        n += kind(i).n;
    return n;
}
// The softmax's and dS's rules for a wave of MR M tiles.
template<int MR>
struct rule
{
    static constexpr int NE = 8 * MR;  // the wave's elements
    // The MFMAs of a fragment that have the one share: all but the last, the step's and the DMA round's.
    static constexpr int shares(int i)
    {
        return MR - 1 - has_step(i) - has_dma(i);
    }
    // The fragments of exps and those of dS. From S's last MR fragments on the loop has: 4 fragments of exps (N tile
    // 0's of the first MR - 1 M tiles), the 3 of N tile 1's fma, 1 of exps, 2 of dS (N tile 0's of those M tiles),
    // 3 of exps (with the one before: N tile 1's of those M tiles), 2 of dS (N tile 1's), the E_LAST of the last M
    // tile's 8 exps, 2 of its dS.
    static constexpr int I_EXP = NF_S - MR, E_LAST = (8 + MR - 2) / (MR - 1);
    static constexpr int I_MUL0 = I_EXP + 8, I_MUL1 = I_EXP + 13, I_MUL_LAST = I_EXP + 15 + E_LAST;
    static constexpr bool of_exps(int i)
    {
        const int k = i - I_EXP;
        return (k >= 0 && k < 4) || k == 7 || (k >= 10 && k < 13) || (k >= 15 && k < 15 + E_LAST);
    }
    // The VALU a fragment has room for: none if it is of exps; else two behind each MFMA with the one share and what
    // the reads and a scalar leave of the two places behind the last MFMA (none in front of a branch).
    static constexpr int room(int i)
    {
        if(of_exps(i))
            return 0;
        const bool branch = i == I_P - 1 || i == NL - 1;
        return 2 * shares(i) + (branch ? 0 : 2 - reads(i) - (scalars(i).n > shares(i) ? 1 : 0));
    }
    // Of the n items of a kind that ride from fragment a on, each fragment taking as many as it has room for beside
    // the USED ones of the kinds in front of it: those behind fragment i.
    template<class U>
    static constexpr span fill(int a, int n, int i, U used)
    {
        int first = 0;
        for(int j = a; j < i && first < n; j++)
            first += clamp(room(j) - used(j), 0, n - first);
        return {first, i < a ? 0 : clamp(room(i) - used(i), 0, n - first)};
    }
    // dS of the tile before to bf16, pair by pair: from S's fragment 2 on.
    static constexpr span packs(int i)
    {
        return fill(2, NE / 2, i, none);
    }
    // S scale - LSE: from S's fragment 6 on (four MFMAs behind the first C tile's last), in what the packs leave.
    static constexpr span fmas(int i)
    {
        return fill(6, NE, i, [](int j) { return packs(j).n; });
    }
    // The exps: one behind each MFMA with the one share of the fragments of exps.
    static constexpr span exps(int i)
    {
        int first = 0;
        for(int j = 0; j < i; j++)
            first += of_exps(j) ? shares(j) : 0;
        return {first, of_exps(i) ? clamp(NE - first, 0, shares(i)) : 0};
    }
    // dS = P (dP - D): N tile 0's of the first MR - 1 M tiles, N tile 1's of those, the last M tile's, each from
    // its fragment on.
    static constexpr span muls(int i)
    {
        const span a = fill(I_MUL0, NE / 2 - 4, i, none), b = fill(I_MUL1, NE / 2 - 4, i, none), c = fill(I_MUL_LAST, 8, i, none);
        return a.n ? a : b.n ? span{NE / 2 - 4 + b.first, b.n} : span{NE - 8 + c.first, c.n};
    }
    // The LDS addresses a tile on. K nope's and K rope's: behind the last fragment of fma, in what they leave. K^T's
    // two and V's: from dQ's fragment 9 on (behind K^T's last read), in what dS leaves.
    static constexpr span addrs(int i)
    {
        const span a = fill(I_EXP + 6, 2, i, [](int j) { return fmas(j).n; });
        const span b = fill(I_Q + 9, 3, i, [](int j) { return muls(j).n; });
        return a.n ? a : span{2 + b.first, b.n};
    }
};
}  // namespace fmha_bwd_d192_v128_dq_sched

// dQ pass of the 4-wave backward, one wave per SIMD, on the recipe of the dK / dV pass's loop: wave w
// owns MR M tiles (16 MR queries from qb0 + 16 MR w; MR is 4, or 3 in a sequence's last blocks); their Q
// and dO (A of S and dP) stay in registers (Q's first 128 dims in AGPRs), LSE and D with them, dQ in AGPRs.
// Per tile of 32 keys (K nope, K rope and V go global -> LDS by DMA into 4 buffers) an iteration is
// 32 fragments of MR MFMAs, one per M tile; what rides behind each is in fmha_bwd_d192_v128_dq_sched. With
// MR = 4:
//   barrier             every wave is past its reads of tile ts - 2, tile ts + 1 is whole
//   S(ts)      48 MFMA  the 6 DMA rounds of tile ts + 2 into tile ts - 2's buffer, dS of tile ts - 1
//                       to bf16, the loop's scalars, S scale - LSE and the exps of the C tiles done
//                       mid block
//   dP(ts)     32 MFMA  the other S scale - LSE, exps, the first of dS = P (dP - D) in fp32
//   dQ(ts - 1) 48 MFMA  the other exps and dS, the LDS addresses a tile on
// S and dP go N tile by N tile (a B fragment serves MR consecutive MFMAs, one per M tile), so
// half their C tiles are done mid block. dQ runs a tile behind so that the VALU work of dS has
// dP's and dQ's MFMAs to go behind. No MFMA waits for LDS: the B operands are in rings of three,
// and behind a fragment's last MFMA its entry is refilled with the fragment three ahead.
// dS = P (dP - D): dP's accumulators start at -D.
template<int MR>
__device__ __attribute__((always_inline)) inline void fmha_bwd_d192_v128_dq_body(
    const fmha_bwd_d192_v128_kargs& k, fmha_bwd_d192_v128::bf16_t* smem, const int lane, const int w, const int h, const int hk, const int qb0,
    unsigned long long bnd_q, unsigned long long bnd_k, fmha_bwd_d192_v128::f32x4_t (&z)[4][12])
{
    using namespace fmha_bwd_d192_v128_dq;
    using fmha_bwd_d192_v128::bf16_t;
    using fmha_bwd_d192_v128::bvec;
    using fmha_bwd_d192_v128::f32x4_t;
    using opus::number;
    using opus::operator""_I;
    using u32x2_t = fmha_bwd_d192_v128::u32x2_t;
    using u32x4_t = fmha_bwd_d192_v128::u32x4_t;
    constexpr int WQ = 16 * MR, BQ = NW * WQ, NT = KC / 16;  // this body's queries a wave, a block
    constexpr int KS = D_QK / 32, KS_N = D_NOPE / 32, KSV = D_V / 32, NG = D_QK / 16, NGN = D_NOPE / 16;
    // dQ, [M tile][N tile of 16 columns], in AGPRs: the kernel's zeros.
    static_assert(WQ <= fmha_bwd_d192_v128_dq::WQ && NG == 12, "the kernel's zeroed tiles, the staging");
    f32x4_t acc[MR][NG];
    // The AGPR file: dQ's accumulators at a[0 ..) (tile (m, n) at a[4 (12 m + n)], where the kernel
    // put its zeros), Q's fragments behind the 48 tiles.
    constexpr int A_ACC = 0, A_AQ = 4 * 4 * NG;
    static_assert(A_AQ + 4 * MR * KS_N <= 256, "the AGPR file");
#pragma unroll
    for(int m = 0; m < MR; m++)
#pragma unroll
        for(int n = 0; n < NG; n++)
            acc[m][n] = z[m][n];

    // The tensors' bases at this head.
    const bf16_t* q_h  = reinterpret_cast<const bf16_t*>(k.ptr_q) + static_cast<long long>(h) * k.stride_q_h;
    const bf16_t* do_h = reinterpret_cast<const bf16_t*>(k.ptr_do) + static_cast<long long>(h) * k.stride_do_h;
    const bf16_t* kn_h = reinterpret_cast<const bf16_t*>(k.ptr_k) + static_cast<long long>(hk) * k.stride_k_h;
    const bf16_t* v_h  = reinterpret_cast<const bf16_t*>(k.ptr_v) + static_cast<long long>(hk) * k.stride_v_h;
    bf16_t* dq_h       = reinterpret_cast<bf16_t*>(k.ptr_dq) + static_cast<long long>(h) * k.stride_dq_h;
    const float* lse_h = k.ptr_lse + static_cast<long long>(h) * k.stride_lse_h;
    const float* d_h   = k.ptr_d + static_cast<long long>(h) * k.stride_lse_h;

    auto make_mma = [](auto e_m, auto e_n, auto e_k) {
        return opus::make_tiled_mma<bf16_t, bf16_t, float>(
            opus::seq<decltype(e_m)::value, decltype(e_n)::value, decltype(e_k)::value>{},
            opus::seq<1, 1, 1>{},
            opus::seq<16, 16, 32>{},
            opus::mfma_adaptor_swap_ab{});
    };
    auto mma_s  = make_mma(number<MR>{}, number<NT>{}, number<KS>{});   // Q's A layout: [M tile][K step][8]
    auto mma_dp = make_mma(number<MR>{}, number<NT>{}, number<KSV>{});  // dO's A layout
    auto mma_k1 = make_mma(number<MR>{}, number<NT>{}, number<1>{});
    auto mma1   = make_mma(number<1>{}, number<1>{}, number<1>{});
    const auto c_ab = opus::make_tuple(0_I, lane % mma_k1.grpm_a, 0_I, lane / mma_k1.grpm_a);
    constexpr int NS = decltype(mma_k1)::tile_c_len;
    static_assert(NS == MR * NT * 4 && NG % 3 == 0 && KS_N == 4, "the fragments below");
    auto rec = [](long long elems, int elem_bytes) -> unsigned int {
        const long long b = elems > 0 ? elems * elem_bytes : 0;
        return b >= 0xffffffffLL ? 0xffffffffu : static_cast<unsigned int>(b);
    };
    // A scalar here: the op that makes it stays in front of this place.
    auto here  = [](auto& x) { asm volatile("" : "+s"(x)); };
    auto fence = [] { __builtin_amdgcn_sched_barrier(0); };
    // A wait for this wave's LDS reads but the last n, a fence behind: in front of the loop and of the tail.
    auto lds_wait = [&](auto n) { opus::s_waitcnt_lgkmcnt(n); fence(); };
    auto barrier = [&] {
        fence();
        __builtin_amdgcn_s_barrier();
        fence();
    };

    float sc2 = k.softmax_scale * fmha_bwd_d192_v128::LOG2_E;
    auto lds_of = [&](int b) { return fmha_bwd_d192_v128::lds_addr(smem + b); };

    // DMA: a tile is 9 loads of 1 KB of K nope, 5 of K rope, 9 of V (slot s of a region is row
    // s / SPR, chunk s % SPR; the padding chunks read out of bounds, as 0). In round xr = 0 .. 5
    // wave w issues load w + NW xr (wave 3's last repeats its fifth). Rounds 0, 1 load K nope, 4, 5 V;
    // round 2 is K nope for wave 0 and K rope for the others, round 3 K rope for waves 0, 1 and V
    // for 2, 3: four descriptors, each pointing at the tile to load.
    constexpr int SPR_N = LD_KN * 2 / 16, SPR_R = LD_KR * 2 / 16, SPR_V = LD_V * 2 / 16;
    constexpr int J_N = KC * SPR_N / 64, J_R = KC * SPR_R / 64, J_V = KC * SPR_V / 64, XR = 6;
    static_assert(KC * SPR_N % 64 == 0 && KC * SPR_R % 64 == 0 && KC * SPR_V % 64 == 0 && J_N == 9 && J_R == 5 && J_V == 9 && NW == 4,
                  "the rounds below");
    constexpr int OOB = 1 << 30;
    // The lanes' byte offsets of load jj of a region (jj wave-uniform; the divisors are constants).
    auto os_n = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_N;
        const int in = -static_cast<int>(c < D_NOPE / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_N * (k.stride_k_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_r = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_R;
        const int in = -static_cast<int>(c < D_ROT / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_R * (k.stride_k_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    auto os_v = [&](int jj) {
        const unsigned int sl = 64u * jj + lane, c = sl % SPR_V;
        const int in = -static_cast<int>(c < D_V / 8);  // all ones in a chunk of the row, 0 in its padding
        return (static_cast<int>(sl / SPR_V * (k.stride_v_n * 2) + c * 16) & in) | (OOB & ~in);
    };
    // readfirstlane: the compiler makes these selects in VALU, with the lanes' offsets'.
    auto u_lds = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int j5 = w == 3 ? 5 : w + 6;
    int x_lds[XR], x_os[XR];
    x_lds[0] = u_lds(OFF_KN * 2 + w * 1024);
    x_os[0]  = os_n(w);
    x_lds[1] = u_lds(OFF_KN * 2 + (w + 4) * 1024);
    x_os[1]  = os_n(w + 4);
    x_lds[2] = u_lds(w == 0 ? OFF_KN * 2 + 8 * 1024 : OFF_KR * 2 + (w - 1) * 1024);
    x_os[2]  = w == 0 ? os_n(8) : os_r(w - 1);
    x_lds[3] = u_lds(w < 2 ? OFF_KR * 2 + (w + 3) * 1024 : OFF_V * 2 + (w - 2) * 1024);
    x_os[3]  = w < 2 ? os_r(w + 3) : os_v(w - 2);
    x_lds[4] = u_lds(OFF_V * 2 + (w + 2) * 1024);
    x_os[4]  = os_v(w + 2);
    x_lds[5] = u_lds(OFF_V * 2 + j5 * 1024);
    x_os[5]  = os_v(j5);

    // Fragments of one K step or N tile, at a per-lane LDS address plus an immediate.
    auto lay_b     = [&](auto ld, const auto& c) { return opus::partition_layout_b<8>(mma_k1, opus::make_tuple(ld, 1_I), c); };
    const auto c_0 = opus::make_tuple(0_I, 0_I, 0_I, 0_I);
    const auto of_kn = opus::layout_to_offsets<8>(lay_b(number<LD_KN>{}, c_0));
    const auto of_kr = opus::layout_to_offsets<8>(lay_b(number<LD_KR>{}, c_0));
    const auto of_v  = opus::layout_to_offsets<8>(lay_b(number<LD_V>{}, c_0));
    int ob_kn  = (opus::layout_to_offsets<8>(lay_b(number<LD_KN>{}, c_ab))[0] - of_kn[0]) * 2;
    int ob_kr  = (opus::layout_to_offsets<8>(lay_b(number<LD_KR>{}, c_ab))[0] - of_kr[0]) * 2;
    int ob_v   = (opus::layout_to_offsets<8>(lay_b(number<LD_V>{}, c_ab))[0] - of_v[0]) * 2;
    int ob_tkn = fmha_bwd_d192_v128::tr_c_order_lane<LD_KN>(lane) * 2;
    int ob_tkr = fmha_bwd_d192_v128::tr_c_order_lane<LD_KR>(lane) * 2;
    // K^T's per-lane addresses are V's plus cv_n, cv_r.
    int cv_n = (OFF_KN - OFF_V) * 2 + ob_tkn - ob_v, cv_r = (OFF_KR - OFF_V) * 2 + ob_tkr - ob_v;
    auto at   = [](unsigned int a) { return opus::make_smem(reinterpret_cast<bf16_t*>(static_cast<__UINTPTR_TYPE__>(a))); };
    auto addr = [&](int b, int off, int ob) {
        unsigned int sb = fmha_bwd_d192_v128::lds_addr(smem + b + off);
        unsigned int a = sb + static_cast<unsigned int>(ob);
        return a;
    };
    // Tile 0's K and V, and K^T: the loop's first dQ block is of no tile (its dS is 0, but what it
    // reads must be finite) and reads tile 0's. At the end of an iteration K^T's addresses are
    // taken from V's, then V's go a tile on.
    unsigned int a_kn = addr(0, OFF_KN, ob_kn), a_kr = addr(0, OFF_KR, ob_kr), a_v = addr(0, OFF_V, ob_v);
    unsigned int a_tkn = addr(0, OFF_KN, ob_tkn), a_tkr = addr(0, OFF_KR, ob_tkr);
    auto rd_k = [&](int tn, int ks) -> bvec<8> {
        return ks < KS_N ? at(a_kn).template load<8>(of_kn[tn] + ks * 32) : at(a_kr).template load<8>(of_kr[tn] + (ks - KS_N) * 32);
    };
    auto rd_v = [&](int tn, int ks) -> bvec<8> { return at(a_v).template load<8>(of_v[tn] + ks * 32); };
    // One transposed read of K's N tile gn (16 columns), half hh of the tile's 32 keys.
    auto rd_t = [&](int gn, int hh) {
        return gn < NGN ? fmha_bwd_d192_v128::tr_read(at(a_tkn), gn * 16 + hh * 16 * LD_KN)
                        : fmha_bwd_d192_v128::tr_read(at(a_tkr), (gn - NGN) * 16 + hh * 16 * LD_KR);
    };
    auto b_tr = [](const u32x2_t (&hv)[2]) { return u32x4_t{hv[0][0], hv[0][1], hv[1][0], hv[1][1]}; };

    float lse2[MR], dd[MR];        // this wave's LSE log2 e and D, per M tile
    f32x4_t nd4[MR];               // -D four times over: where dP's accumulators start
    u32x4_t aq[MR * KS_N];         // Q's first 128 dims, [M tile][K step], in AGPRs
    bvec<8> bq[3], bv[3];          // B fragments of S and dP: the one in use and the next two
    u32x2_t tb[3][2];              // B of dQ: this N tile's two reads and the next two's
    opus::vector_t<float, NS> v_sp, v_dp;
    opus::vector_t<bf16_t, MR * 8> a_ds;
    // The VGPR file: S's C tile tl at v[V_SP + 4 tl], dP's at v[V_DP + 4 tl], at the top of the 240.
    constexpr int V_DP = 240 - NS, V_SP = V_DP - NS;
    // dS's bf16 (the A of dQ) below them, the LDS addresses (a_kn, a_kr, a_v, a_tkn, a_tkr) below that.
    constexpr int V_ADS = V_SP - 4 * MR, V_ADR = V_ADS - 5;
    // One MFMA of S or dP onto C tile tl, its result in the tile's registers (reg): from 0, or from c0
    // (step 0), or in place.
    auto mmp = [&](auto reg, const bvec<8>& fa, const bvec<8>& fb, auto& vacc, int tl, int step, const f32x4_t* c0 = nullptr) __attribute__((always_inline)) {
        f32x4_t cc = c0 ? *c0 : f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        if(step != 0)
#pragma unroll
            for(int p_ = 0; p_ < 4; p_++)
                cc[p_] = vacc[tl * 4 + p_];
        cc = fmha_bwd_d192_v128::pin_vgpr<decltype(reg)::value>(mma1(fa, fb, cc));
#pragma unroll
        for(int p_ = 0; p_ < 4; p_++)
            vacc[tl * 4 + p_] = cc[p_];
    };
    // The loop's VALU is C++ with its results pinned. The compiler counts the wait states between an
    // MFMA and a VALU read of its result in instructions, not cycles: the schedule keeps those reads
    // far enough behind the MFMA that it adds no s_nop.
    auto pack = [&](int j) __attribute__((always_inline)) {
        opus::vector_t<bf16_t, 2> h2;
        h2 = __builtin_convertvector((opus::vector_t<float, 2>{v_dp[2 * j], v_dp[2 * j + 1]}), opus::vector_t<bf16_t, 2>);
        a_ds[2 * j]     = h2[0];
        a_ds[2 * j + 1] = h2[1];
    };
    // S scale - LSE of one element, the exp (P), dS = P (dP - D): the instructions the compiler
    // would make of the expressions.

    // The bounds are in (the kernel waited for them).
    const int q0 = static_cast<int>(bnd_q), k0 = static_cast<int>(bnd_k);
    const int seqlen_q = static_cast<int>(bnd_q >> 32) - q0, seqlen_k = static_cast<int>(bnd_k >> 32) - k0;
    // A block past the sequence's queries has nothing to do (no early return, as in the dK / dV body).
    if(qb0 < seqlen_q)
    {
    // Bottom-right causal: query qp sees keys kv <= qp + causal_offset. The workgroup's tiles are
    // those its last query sees, 0 .. T - 1; if it sees none the loop runs tile 0, all masked.
    const int causal_offset = seqlen_k - seqlen_q;
    const int q_last        = (qb0 + BQ < seqlen_q ? qb0 + BQ : seqlen_q) - 1;
    int kv_end              = seqlen_k;
    if(k.causal && q_last + causal_offset + 1 < kv_end)
        kv_end = q_last + causal_offset + 1;
    const int T = kv_end > 0 ? (kv_end + KC - 1) / KC : 0;
    // This wave's 16 MR queries; rows past seqlen_q read as 0.
    const int qw0    = qb0 + w * WQ;
    const int rows_q = seqlen_q - qw0 < WQ ? (seqlen_q - qw0 > 0 ? seqlen_q - qw0 : 0) : WQ;

    // The tiles this wave sees in full: [0, n_full). Rows past seqlen_q count as rows: Q, dO, LSE
    // and D read as 0 there, so P is 1 and dP - D 0, and they add nothing whatever they see.
    int n_full = seqlen_k / KC;
    if(k.causal)
    {
        const int lim = qw0 + causal_offset + 1 > 0 ? (qw0 + causal_offset + 1) / KC : 0;
        n_full        = lim < n_full ? lim : n_full;
    }
    here(n_full);
    fence();

    // LSE and D of the wave's queries: two small loads per M tile.
    {
        const float* lse_p = lse_h + (q0 + qw0);
        const float* d_p   = d_h + (q0 + qw0);
        unsigned int ld_sz = static_cast<unsigned int>(__builtin_amdgcn_readfirstlane(rows_q * 4));
        auto g_lse = opus::make_gmem(lse_p, ld_sz);
        auto g_d   = opus::make_gmem(d_p, ld_sz);
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
        {
            lse2[tm] = g_lse.template load<1>(16 * tm + lane % 16)[0];
            dd[tm]   = g_d.template load<1>(16 * tm + lane % 16)[0];
        }
    }
    const char* kn_p = reinterpret_cast<const char*>(kn_h + static_cast<long long>(k0) * k.stride_k_n);
    const char* kr_p = kn_p + D_NOPE * 2;  // the rope part of the same rows
    const char* v_p  = reinterpret_cast<const char*>(v_h + static_cast<long long>(k0) * k.stride_v_n);
    const int rb_n = __builtin_amdgcn_readfirstlane(k.stride_k_n * 2);
    const int rb_v = __builtin_amdgcn_readfirstlane(k.stride_v_n * 2);
    const int x_rb[4] = {rb_n, rb_n, __builtin_amdgcn_readfirstlane(w < 2 ? rb_n : rb_v), rb_v};
    const char* x_src[4] = {kn_p, w == 0 ? kn_p : kr_p, w < 2 ? kr_p : v_p, v_p};
    int x_kb[4];
    fmha_bwd_d192_v128::i32x4_t x_rs[4];
#pragma unroll
    for(int d = 0; d < 4; d++)
    {
        x_kb[d] = KC * x_rb[d];
        x_rs[d] = fmha_bwd_d192_v128::dma_rsrc(x_src[d], 0);
    }
    auto x_sizes = [&](int t) {
        int rows = seqlen_k - t * KC;
        rows = rows < KC ? rows : KC;
        rows = __builtin_amdgcn_readfirstlane(rows > 0 ? rows : 0);  // the clamp is a VALU med3: one scalar for the products
#pragma unroll
        for(int d = 0; d < 4; d++)
        {
            int nb;
            nb = rows * x_rb[d];
            x_rs[d] = fmha_bwd_d192_v128::i32x4_t{x_rs[d][0], x_rs[d][1], nb, opus::buffer_default_config()};
        }
    };
    auto x_adv = [&](int d) {
        auto& r = x_rs[d];
        unsigned long long a =
            static_cast<unsigned int>(r[0]) | static_cast<unsigned long long>(static_cast<unsigned int>(r[1])) << 32;
        a += static_cast<unsigned long long>(static_cast<unsigned int>(x_kb[d]));
        r = fmha_bwd_d192_v128::i32x4_t{static_cast<int>(a), static_cast<int>(a >> 32), r[2], r[3]};
    };
    auto x_go = [&](int xr, unsigned int lb) {
        const int d = xr < 2 ? 0 : (xr - 1 < 3 ? xr - 1 : 3);
        fmha_bwd_d192_v128::dma_round<16>(x_rs[d], lb + x_lds[xr], x_os[xr]);
    };

    // Tiles 0 and 1 to LDS (buffers 0 and 1).
    {
        unsigned int lb0 = lds_of(0), lb1 = lds_of(SZ_BUF);
        x_sizes(0);
#pragma unroll
        for(int xr = 0; xr < XR; xr++)
            x_go(xr, lb0);
#pragma unroll
        for(int d = 0; d < 4; d++)
            x_adv(d);
        x_sizes(1);
#pragma unroll
        for(int xr = 0; xr < XR; xr++)
            x_go(xr, lb1);
        // Descriptor 0 goes a tile on past its rounds, the others in front of theirs.
        x_adv(0);
        x_sizes(2);
    }
    // This wave's queries as the A operand of S and dP, in registers.
    auto g_q  = opus::make_gmem(q_h + static_cast<long long>(q0 + qw0) * k.stride_q_n, rec(static_cast<long long>(rows_q) * k.stride_q_n, 2));
    auto g_do = opus::make_gmem(do_h + static_cast<long long>(q0 + qw0) * k.stride_do_n, rec(static_cast<long long>(rows_q) * k.stride_do_n, 2));
    const auto v_qw  = g_q.template load<8>(opus::partition_layout_a<8>(mma_s, opus::make_tuple(k.stride_q_n, 1_I), c_ab));
    const auto v_dow = g_do.template load<8>(opus::partition_layout_a<8>(mma_dp, opus::make_tuple(k.stride_do_n, 1_I), c_ab));
    {
        // Loop-carried scalars, at the top of the iteration of S tile ts: lbc, lbp the LDS address
        // of tile ts - 1's buffer and of tile ts - 2's (the DMA's target, tile ts + 2 goes there);
        // dlt the LDS addresses' step from tile ts - 1's buffer to tile ts's (from tile ts's to the
        // next past the scalar C_DLT); rc tile ts's buffer in the ring (0 .. 3), dn the step from it to
        // the next (-3 buffers past the last, without flags: single scalars go anywhere);
        // ra = seqlen_k - (ts + 4) KC (negative: tile ts + 3 is not a full one), nf = n_full - 1
        // - ts (negative: mask), z = ts - T.
        constexpr int SZB = SZ_BUF * 2;
        static_assert(NBUF == 4, "the ring's wrap below");
        int rc = 0, dn = 0;
        int ring = -NBUF * SZB, step = SZB;  // in registers: with a literal an instruction is two words
        int dlt = -(NBUF - 1) * SZB;
        unsigned int lbc = lds_of((NBUF - 1) * SZ_BUF), lbp = lds_of((NBUF - 2) * SZ_BUF);
        int ra = seqlen_k - 4 * KC, nf = n_full - 1, z = -T;

        namespace sch = fmha_bwd_d192_v128_dq_sched;
        using rule    = sch::rule<MR>;
        using sch::reads;
        // The loop's fragments in order: S's, dP's, dQ's. sch's rules give every element its three steps and every
        // pair its pack, the loop its scalars and addresses.
        constexpr int NF_S = NT * KS, NF_P = NT * KSV, I_P = NF_S, I_Q = NF_S + NF_P, NL = I_Q + NG;
        static_assert(rule::NE == NS && sch::NF_S == NF_S && sch::NF_P == NF_P && sch::NF_Q == NG && sch::NL == NL &&
                          sch::has_dma(XR - 1) && !sch::has_dma(XR),
                      "the loop of the rules");
        static_assert(sch::total(sch::scalars) == sch::C_N && sch::total(rule::packs) == NS / 2 && sch::total(rule::fmas) == NS &&
                          sch::total(rule::exps) == NS && sch::total(rule::muls) == NS && sch::total(rule::addrs) == 5,
                      "the loop's work");
        constexpr unsigned int S_EXPD = sch::exp_done(MR, rule::exps(I_P).first);

        // The GEMMs, a fragment at a time: its MR MFMAs, one per M tile, on its B operand in the ring; behind(F) is
        // what rides behind them, as the per_step of the MLA kernels' compute_qk; a fence ends the fragment.
        // S of tile ts: fragment f is N tile f / 6, K step f % 6.
        auto gemm_s = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NF_S>([&](auto F) {
                constexpr int f = decltype(F)::value, tn = f / KS, ks = f % KS;
                opus::static_for<MR>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    bvec<8> fa;
                    if constexpr(ks < KS_N)
                        fa = __builtin_bit_cast(bvec<8>, aq[j * KS_N + ks]);
                    else
#pragma unroll
                        for(int x = 0; x < 8; x++)
                            fa[x] = v_qw[(j * KS + ks) * 8 + x];
                    mmp(number<V_SP + 4 * (NT * j + tn)>{}, fa, bq[f % 3], v_sp, NT * j + tn, ks == 0 ? 0 : 1);
                });
                behind(F);
                fence();
            });
        };
        // dP of tile ts: fragment f is N tile f / 4, K step f % 4; the accumulators start at -D.
        auto gemm_dp = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NF_P>([&](auto F) {
                constexpr int f = decltype(F)::value, tn = f / KSV, ks = f % KSV;
                opus::static_for<MR>([&](auto J) {
                    constexpr int j = decltype(J)::value;
                    bvec<8> fa;
#pragma unroll
                    for(int x = 0; x < 8; x++)
                        fa[x] = v_dow[(j * KSV + ks) * 8 + x];
                    mmp(number<V_DP + 4 * (NT * j + tn)>{}, fa, bv[f % 3], v_dp, NT * j + tn, ks == 0 ? 0 : 1,
                        ks == 0 ? &nd4[j] : nullptr);
                });
                behind(F);
                fence();
            });
        };
        // dQ of the tile before (of the last tile in the tail): a fragment is an N tile of K^T, 16 of dQ's columns.
        auto gemm_dq = [&](auto&& behind) __attribute__((always_inline)) {
            opus::static_for<NG>([&](auto G) {
                constexpr int gn = decltype(G)::value;
                opus::static_for<MR>([&](auto M) {
                    constexpr int m = decltype(M)::value;
                    acc[m][gn] = fmha_bwd_d192_v128::pin_agpr<A_ACC + 4 * (m * NG + gn)>(
                        mma1(__builtin_bit_cast(bvec<8>, fmha_bwd_d192_v128::frag8(a_ds, m)),
                             __builtin_bit_cast(bvec<8>, b_tr(tb[gn % 3])), acc[m][gn]));
                });
                behind(G);
                fence();
            });
        };

        // What rides behind the loop's fragment i: a chunk for each kind of work. A chunk takes the kind's items
        // first .. first + n - 1, as the rule of the kind gives them to the fragment; each item's result goes to
        // its own register.
        // dS of the tile before to bf16, the pairs N tile by N tile (pair j: elements 2 j, 2 j + 1): the A operand of
        // the dQ block.
        auto pack_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = rule::packs(decltype(I)::value).first, n = rule::packs(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int j = sch::elem(MR, 2 * (first + decltype(X)::value)) / 2;
                    auto h2 = __builtin_convertvector((opus::vector_t<float, 2>{v_dp[2 * j], v_dp[2 * j + 1]}), opus::vector_t<bf16_t, 2>);
                    h2 = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, fmha_bwd_d192_v128::pin_vgpr<V_ADS + j>(__builtin_bit_cast(opus::u32_t, h2)));
                    a_ds[2 * j]     = h2[0];
                    a_ds[2 * j + 1] = h2[1];
                });
        };
        // The softmax, in place: S scale - LSE, the elements N tile by N tile.
        auto scale_sub_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = rule::fmas(decltype(I)::value).first, n = rule::fmas(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem(MR, first + decltype(X)::value);
                    v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(__builtin_fmaf(v_sp[e], sc2, -lse2[e / (NS / MR)]));
                });
        };
        // Its exp, P: the last M tile's elements behind the others'.
        auto exp_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = rule::exps(decltype(I)::value).first, n = rule::exps(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem_exp(MR, first + decltype(X)::value);
                    v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(__builtin_amdgcn_exp2f(v_sp[e]));
                });
        };
        // dS = P (dP - D) in fp32, in dP's place: the elements in the exps' order.
        auto ds_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = rule::muls(decltype(I)::value).first, n = rule::muls(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int e = sch::elem_exp(MR, first + decltype(X)::value);
                    v_dp[e] = fmha_bwd_d192_v128::pin_vgpr<V_DP + e>(v_dp[e] * v_sp[e]);
                });
        };
        // The LDS addresses a tile on, in their order: K nope's, K rope's, then K^T's two from V's, then V's.
        auto addr_chunk = [&](auto I) __attribute__((always_inline)) {
            constexpr int first = rule::addrs(decltype(I)::value).first, n = rule::addrs(decltype(I)::value).n;
            if constexpr(n > 0)
                opus::static_for<n>([&](auto X) {
                    constexpr int a = first + decltype(X)::value;
                    if constexpr(a == 0)
                        a_kn = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 0>(a_kn + static_cast<unsigned int>(dlt));
                    else if constexpr(a == 1)
                        a_kr = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 1>(a_kr + static_cast<unsigned int>(dlt));
                    else if constexpr(a == 2)
                        a_tkn = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 3>(a_v + static_cast<unsigned int>(cv_n));
                    else if constexpr(a == 3)
                        a_tkr = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 4>(a_v + static_cast<unsigned int>(cv_r));
                    else
                        a_v = fmha_bwd_d192_v128::pin_vgpr<V_ADR + 2>(a_v + static_cast<unsigned int>(dlt));
                });
        };
        // The chunks of the loop's fragment i, in the order they issue: the VALU fill the MFMAs' shares in the code's
        // order, the exps come last.
        auto softmax_chunk = [&](auto I) __attribute__((always_inline)) {
            pack_chunk(I);
            scale_sub_chunk(I);
            ds_chunk(I);
            addr_chunk(I);
            exp_chunk(I);
        };
        // The schedule of a fragment, one sched_mfma per MFMA. With STEP its first MFMA takes a DMA descriptor's
        // step alone, with DMA its second a DMA round. The last takes the wait, the READS LDS reads and what VALU
        // fits beside them. Every other MFMA has the same share: two VALU and an exp, of which a fragment gives it
        // the one or the other. Of the fragment's SC scalars one goes behind each of these, from the first, and if
        // there are more behind the last, in place of a VALU.
        auto sched_chunk = [&](auto SC, auto STEP, auto DMA, auto READS) __attribute__((always_inline)) {
            constexpr int n_sc = decltype(SC)::value, n_step = decltype(STEP)::value, n_dma = decltype(DMA)::value;
            constexpr int n_rd = decltype(READS)::value;
            constexpr int s_last = n_sc > MR - 1 - n_step - n_dma ? 1 : 0;
            static_assert(n_sc <= MR - n_step - n_dma && n_rd + s_last <= 2, "a fragment's MFMAs");
            opus::static_for<MR>([&](auto J) {
                constexpr int j = decltype(J)::value;
                // A scalar is two instructions for the schedule: the op and its anchor.
                constexpr int s2 = 2 * (j - n_step - (n_dma && j > 1) < n_sc);
                if constexpr(n_step && j == 0)
                    fmha_bwd_d192_v128::sched_mfma<2, 0>();
                else if constexpr(n_dma && j == 1)
                    fmha_bwd_d192_v128::sched_mfma<1, 0, 0, 0, 1>();
                else if constexpr(j == MR - 1)
                    fmha_bwd_d192_v128::sched_mfma<1 + 2 * s_last, 2 - n_rd - s_last, 0, n_rd>();
                else
                    fmha_bwd_d192_v128::sched_mfma<s2, 2, 1>();
            });
        };

        // S of tile ts. Behind fragment f: 0 .. 5 a DMA round of tile ts + 2 (0 .. 3 a descriptor's step too); 0, 1
        // and 4 .. 10 the loop's scalars; from 2 on dS of the tile before to bf16; from 6 on S scale - LSE of N
        // tile 0, in what the packs leave; the last MR fragments exps. Returns a negative number if the sizes of
        // tile ts + 3 are to be set or S is to be masked (rare_fix).
        auto s_block = [&]() __attribute__((always_inline)) {
            int rare = 0;
            // The loop's scalar c, anchored where it stands.
            auto scalar = [&](auto C) __attribute__((always_inline)) {
                constexpr int c = decltype(C)::value;
                // The ring a buffer on: rc = (rc + 1) & 3, dn = rc == 0 ? -3 buffers : 1 buffer.
                if constexpr(c == sch::C_RC_INC)
                    { rc += 1; here(rc); }
                else if constexpr(c == sch::C_RC_AND)
                    { rc &= 3; here(rc); }
                else if constexpr(c == sch::C_DN_SUB)
                    { dn = rc - 1; here(dn); }
                else if constexpr(c == sch::C_DN_SHR)
                    { dn >>= 31; here(dn); }
                else if constexpr(c == sch::C_DN_AND)
                    { dn &= ring; here(dn); }
                // Past the last DMA round: tile ts's buffer, the DMA's next target (the buffer before it: no
                // copy, which would be no instruction of the schedule's), then the address step a tile on.
                else if constexpr(c == sch::C_LBC)
                    { lbc += static_cast<unsigned int>(dlt); here(lbc); }
                else if constexpr(c == sch::C_LBP)
                    { lbp = lbc - static_cast<unsigned int>(dlt); here(lbp); }
                else if constexpr(c == sch::C_DLT)
                    { dlt = dn + step; here(dlt); }
                // The rare cases' sign from the S tile's ra and nf, then the counters on.
                else if constexpr(c == sch::C_RARE)
                    { rare = ra | nf; here(rare); }
                else if constexpr(c == sch::C_Z)
                    { z += 1; here(z); }
                else if constexpr(c == sch::C_RA)
                    { ra -= KC; here(ra); }
                else if constexpr(c == sch::C_NF)
                    { nf -= 1; here(nf); }
            };
            using sch::has_dma;
            using sch::has_step;
            // Behind the first MFMA: a DMA descriptor a tile on (1 .. 3 in front of their rounds, 0 past its), or
            // the fragment's first scalar.
            auto head_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                if constexpr(has_step(f))
                    x_adv(sch::S_ADV[f]);
                else if constexpr(sch::scalars(f).n > 0)
                    scalar(number<sch::scalars(f).first>{});
            };
            // Behind the second MFMA: DMA round f of tile ts + 2.
            auto dma_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                if constexpr(has_dma(f))
                    x_go(f, lbp);
            };
            // The fragment's other scalars: one behind each of the other MFMAs, from the first.
            auto scalar_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value, lead = !has_step(f) && sch::scalars(f).n > 0 ? 1 : 0;
                constexpr int first = sch::scalars(f).first + lead, n = sch::scalars(f).n - lead;
                if constexpr(n > 0)
                    opus::static_for<n>([&](auto X) { scalar(number<first + decltype(X)::value>{}); });
            };
            // Behind the last MFMA: the wait for the next fragment's K, in the place that is free only for a scalar
            // (in flight are the reads behind the fragment before), then the ring's refill; behind the last
            // fragments, V's first three for the dP block.
            auto refill_chunk = [&](auto F) __attribute__((always_inline)) {
                constexpr int f = decltype(F)::value;
                opus::s_waitcnt_lgkmcnt(number<reads(f - 1)>{});
                if constexpr(f + 3 < NF_S)
                    bq[f % 3] = rd_k((f + 3) / KS, (f + 3) % KS);
                else if constexpr(f == NF_S - 3)
                    bv[0] = rd_v(0, 0);
                else if constexpr(f == NF_S - 2)
                {
                    bv[1] = rd_v(0, 1);
                    bv[2] = rd_v(0, 2);
                }
            };
            gemm_s([&](auto F) {
                constexpr int f = decltype(F)::value;
                head_chunk(F);
                dma_chunk(F);
                scalar_chunk(F);
                softmax_chunk(F);
                refill_chunk(F);
                sched_chunk(number<sch::scalars(f).n>{}, number<has_step(f)>{}, number<has_dma(f)>{}, number<reads(f)>{});
            });
            return rare;
        };
        // The rare cases: tile ts + 3 is not a full one (its sizes), this wave does not see tile
        // ts in full: the elements it does not see take P = 0 where the exp is made already,
        // else an S (or S scale - LSE) that exps to 0.
        auto rare_fix = [&](int ts) {
            if(seqlen_k - (ts + 4) * KC < 0)
                x_sizes(ts + 3);
            if(ts >= n_full)
            {
                // Element e is row 16 (e / 8) + lane % 16 of the wave, key 16 (e / 4 % 2) +
                // 4 (lane / 16) + e % 4 of the tile; xm[M tile] is the last key of the tile that
                // row sees less the lane's 4 (lane / 16) (causal: a key is seen if kv <= qp +
                // causal_offset, which is below seqlen_k for a row below seqlen_q).
                const int kv0 = ts * KC;
                int xm[MR];
#pragma unroll
                for(int m = 0; m < MR; m++)
                    xm[m] = (k.causal ? qw0 + causal_offset + 16 * m + lane % 16 : seqlen_k - 1) - kv0 - 4 * (lane / 16);
                // The mask, four elements (one C tile) at a time: four compares into scalar pairs,
                // then the four selects, so that each select stands three instructions behind its
                // compare.
                opus::static_for<NS / 4>([&](auto G) {
                    constexpr int gt = decltype(G)::value, c0 = 16 * (gt % NT), e0 = 4 * gt;
                    // The four compares, then the four selects: each select stands three behind its compare.
                    bool in[4];
#pragma unroll
                    for(int p_ = 0; p_ < 4; p_++)
                        in[p_] = xm[gt / NT] >= c0 + p_;
                    __builtin_amdgcn_sched_barrier(0);
                    opus::static_for<4>([&](auto P) {
                        constexpr int e = e0 + decltype(P)::value;
                        const float m = (S_EXPD >> e & 1) ? 0.0f : -1e30f;
                        v_sp[e] = fmha_bwd_d192_v128::pin_vgpr<V_SP + e>(in[decltype(P)::value] ? v_sp[e] : m);
                    });
                });
            }
        };
        // dP of tile ts. Behind its fragments, where sch::rule has them from S's exps on: the rest of S scale - LSE
        // (three fragments; K's addresses a tile on behind the last), one fragment of exps, N tile 0's dS (two:
        // dP's N tile 0 is done), exps. Behind the last MFMA: the wait for the next fragment's V, then the V
        // ring's refill; behind the last three fragments, K^T's first three N tiles for the dQ block.
        auto dp_block = [&]() __attribute__((always_inline)) {
            gemm_dp([&](auto F) {
                constexpr int f = decltype(F)::value, i = I_P + f;
                softmax_chunk(number<i>{});
                opus::s_waitcnt_lgkmcnt(number<reads(i - 1)>{});
                if constexpr(f + 3 < NF_P)
                    bv[f % 3] = rd_v((f + 3) / KSV, (f + 3) % KSV);
                else
                {
                    tb[f + 3 - NF_P][0] = rd_t(f + 3 - NF_P, 0);
                    tb[f + 3 - NF_P][1] = rd_t(f + 3 - NF_P, 1);
                }
                sched_chunk(number<0>{}, number<0>{}, number<0>{}, number<reads(i)>{});
            });
        };
        // dQ of the tile before; TAIL: of the last tile, behind the loop. In the loop its fragments take, where
        // sch::rule has them: the rest of the exps, N tile 1's dS (two fragments), the last M tile's exps and its dS
        // (two), and from fragment 9 on K^T's and V's addresses a tile on. Behind the last MFMA, the wait for the
        // next N tile of K^T and the K^T ring's refill; the last three read the next tile's first three fragments
        // of K for its S block, and the last waits for this wave's DMA loads too (issued in this iteration's S
        // block: in before the barrier). The tail has the waits and the refills alone.
        auto dq_block = [&](auto TAIL) __attribute__((always_inline)) {
            constexpr bool tail = decltype(TAIL)::value;
            gemm_dq([&](auto G) {
                constexpr int gn = decltype(G)::value, i = I_Q + gn;
                // The reads behind this fragment and behind the one before (the tail's ring ends with K^T).
                constexpr int n_rd = gn + 3 < NG ? 2 : (tail ? 0 : reads(i));
                constexpr int n_before = gn == 0 || gn + 2 < NG ? 2 : (tail ? 0 : reads(i - 1));
                if constexpr(!tail)
                    softmax_chunk(number<i>{});
                if constexpr(!tail && gn == NG - 1)
                    fmha_bwd_d192_v128::wait_loads_reads<n_before>();
                else
                    opus::s_waitcnt_lgkmcnt(number<n_before>{});
                if constexpr(gn + 3 < NG)
                {
                    tb[gn % 3][0] = rd_t(gn + 3, 0);
                    tb[gn % 3][1] = rd_t(gn + 3, 1);
                }
                else if constexpr(!tail && gn == NG - 3)
                    bq[0] = rd_k(0, 0);
                else if constexpr(!tail && gn == NG - 2)
                {
                    bq[1] = rd_k(0, 1);
                    bq[2] = rd_k(0, 2);
                }
                sched_chunk(number<0>{}, number<0>{}, number<0>{}, number<n_rd>{});
            });
        };

        // Q, dO, LSE and D are in registers and this wave's loads of tiles 0 and 1 in LDS (wait_loads: the
        // compiler then waits for none of them inside the loop); every wave's are behind the barrier.
        fmha_bwd_d192_v128::wait_loads();
        opus::static_for<MR * KS_N>([&](auto I) {
            constexpr int i = decltype(I)::value;
            aq[i] = fmha_bwd_d192_v128::pin_agpr<A_AQ + 4 * i>(fmha_bwd_d192_v128::frag8(v_qw, i / KS_N * KS + i % KS_N));
        });
#pragma unroll
        for(int e = 0; e < NS; e++)
        {
            v_sp[e] = 0.0f;  // P and dS of the tile before the first are 0
            v_dp[e] = 0.0f;
        }
#pragma unroll
        for(int e = 0; e < MR * 8; e++)
            a_ds[e] = static_cast<bf16_t>(0.0f);
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
            lse2[tm] *= fmha_bwd_d192_v128::LOG2_E;
#pragma unroll
        for(int tm = 0; tm < MR; tm++)
        {
            float nd = dd[tm];
            nd = -nd;
            nd4[tm] = f32x4_t{nd, nd, nd, nd};
        }
        barrier();
        bq[0] = rd_k(0, 0);
        bq[1] = rd_k(0, 1);
        bq[2] = rd_k(0, 2);
        lds_wait(number<2>{});
        // The loop's code starts on a 64-byte boundary.
        do
        {
            // Every wave is past dQ's reads of tile ts - 2, whose buffer this S block's DMA
            // overwrites, and has its loads of tile ts + 1 in.
            barrier();
            const int rare = s_block();
            if(__builtin_expect(rare < 0, 0))
                rare_fix(z - 1 + T);
            dp_block();
            dq_block(number<0>{});
        } while(__builtin_expect(z < 0, 1));
        // The last tile's dQ: its dS to bf16, K^T's first three N tiles (the ring's head, which in the loop the dP
        // block reads; K^T's addresses the last iteration left on the last tile). The first N tile is in.
#pragma unroll
        for(int jp = 0; jp < NS / 2; jp++)
            pack(jp);
#pragma unroll
        for(int x = 0; x < 6; x++)
            tb[x / 2][x % 2] = rd_t(x / 2, x % 2);
        lds_wait(number<4>{});
        dq_block(number<1>{});
    }
    // The loads still out go to LDS; the accumulators' last MFMAs finish.
    fmha_bwd_d192_v128::wait_loads();
    // The C fragments hold 4 columns of 16 rows per instruction, so each wave transposes its 16 MR
    // rows through LDS (free once every wave's DMA writes are in: the barrier) and stores whole
    // rows, 64 lanes x 8 bf16 on end; rows past seqlen_q fall outside the descriptor and are
    // dropped.
    fmha_bwd_d192_v128::block_sync();
    __builtin_amdgcn_sched_barrier(0);  // as the dK / dV epilogue's
    {
        static_assert(NW * WQ * LD_O <= SMEM_ELEMS && WQ * (D_QK / 8) % 64 == 0, "staging fits; whole instructions");
        bf16_t* s_o = smem + w * WQ * LD_O;
        // C tile (m, n): row 16 m + lane % 16, columns 16 n + 4 (lane / 16) + 0 .. 3.
        const int c_row = lane % 16, c_col = 4 * (lane / 16);
        opus::static_for<MR * NG>([&](auto I) {
            constexpr int i = decltype(I)::value, m = i / NG, n = i % NG;
            *reinterpret_cast<bvec<4>*>(s_o + (16 * m + c_row) * LD_O + 16 * n + c_col) =
                opus::cast<bf16_t>(fmha_bwd_d192_v128::pin_agpr<A_ACC + 4 * i>(acc[m][n]) * k.softmax_scale);
            if constexpr(i % 2 == 1)  // two tiles a fence: their two LDS writes one ds_write2
                __builtin_amdgcn_sched_barrier(0);
        });
        auto g_dq = opus::make_gmem(dq_h + static_cast<long long>(q0 + qw0) * k.stride_dq_n,
                                    rec(static_cast<long long>(rows_q) * k.stride_dq_n, 2));
        constexpr int OC = D_QK / 8;  // 8-element chunks of a row
#pragma unroll
        for(int it = 0; it < WQ * OC / 64; it++)
        {
            const int ch = it * 64 + lane, row = ch / OC, col = ch % OC * 8;
            g_dq.template store<8>(*reinterpret_cast<const bvec<8>*>(s_o + row * LD_O + col), row * k.stride_dq_n + col);
        }
    }
    }
}


// The 4-wave backward: the dK / dV pass and the dQ pass in one launch, a workgroup id being of
// one or the other by its place in the grid. 4 waves, one per SIMD; a pass with no blocks takes
// no workgroups. The kernel reads its arguments (scalar loads, the accumulators' zeros behind
// them: the dQ pass's 48 tiles, of which the dK / dV pass takes 36) and hands over to the pass's
// body. A __device__ function, as the pre-pass's: the extern "C" entries aiter looks up by name
// are in fmha_bwd_hd192_v128_bf16.cu.
__device__ inline __attribute__((always_inline)) void fmha_bwd_d192_v128_body(const fmha_bwd_d192_v128_kargs& k_)
{
    using fmha_bwd_d192_v128::bf16_t;
    using fmha_bwd_d192_v128::f32x4_t;
    constexpr int SMEM = fmha_bwd_d192_v128_dkdv::SMEM_ELEMS > fmha_bwd_d192_v128_dq::SMEM_ELEMS ? fmha_bwd_d192_v128_dkdv::SMEM_ELEMS : fmha_bwd_d192_v128_dq::SMEM_ELEMS;
    static_assert(fmha_bwd_d192_v128_dkdv::BLOCK == fmha_bwd_d192_v128_dq::BLOCK, "one workgroup shape");
    __shared__ __attribute__((aligned(16))) bf16_t smem[SMEM];

    const int tid  = static_cast<int>(opus::thread_id_x());
    const int lane = tid % 64;
    const int w    = __builtin_amdgcn_readfirstlane(tid / 64);
    fmha_bwd_d192_v128_kargs k = k_;

    // The zeros, in the dQ accumulators' AGPRs from here on: values of registers, not constants
    // to make again where they are used.
    f32x4_t z[4][12];
#pragma unroll
    for(int m = 0; m < 4; m++)
#pragma unroll
        for(int n = 0; n < 12; n++)
        {
            z[m][n] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
        }
    // Tile (m, n) of the zeros in a[4 (12 m + n)]: where the dQ pass keeps its accumulators.
    opus::static_for<48>([&](auto I) {
        constexpr int i = decltype(I)::value;
        z[i / 12][i % 12] = fmha_bwd_d192_v128::pin_agpr<4 * i>(z[i / 12][i % 12]);
    });
    // Every argument goes through here, in and out, so that the kernel reads them in one batch
    // of scalar loads with one wait, behind the zeros.
    asm volatile("" : "+s"(k.ptr_q), "+s"(k.ptr_k), "+s"(k.ptr_v), "+s"(k.ptr_do), "+s"(k.ptr_lse), "+s"(k.ptr_d));
    asm volatile("" : "+s"(k.ptr_seqstart_q), "+s"(k.ptr_seqstart_k), "+s"(k.ptr_dq), "+s"(k.ptr_dk), "+s"(k.ptr_dv));
    asm volatile("" : "+s"(k.H), "+s"(k.num_seqs), "+s"(k.n_kv_blocks), "+s"(k.n_q_blocks), "+s"(k.rem_magic), "+s"(k.causal), "+s"(k.softmax_scale));
    asm volatile("" : "+s"(k.stride_q_n), "+s"(k.stride_q_h), "+s"(k.stride_k_n), "+s"(k.stride_k_h), "+s"(k.stride_v_n));
    asm volatile("" : "+s"(k.stride_v_h), "+s"(k.stride_do_n), "+s"(k.stride_do_h), "+s"(k.stride_lse_h), "+s"(k.stride_dq_n), "+s"(k.stride_dq_h));
    asm volatile("" : "+s"(k.stride_dk_n), "+s"(k.stride_dk_h), "+s"(k.stride_dv_n), "+s"(k.stride_dv_h), "+s"(k.div_magic), "+s"(k.div_magic_q), "+s"(k.kv_rows), "+s"(k.kv_magic));

    // The workgroup's place is its place in the grid (fmha_bwd_d192_v128_kargs). With a causal mask
    // (by_group) z is the group and the sequence, y the block of the dK / dV pass, then of the dQ
    // pass; without, z is the sequence, y the group and the block of the dK / dV pass, then of the
    // dQ pass. x is the head in its group, but for a last group of fewer heads, whose ids are packed.
    // readfirstlane: these are scalars, and must stay so for the DMA's descriptors and m0.
    auto sc = [](int x) { return __builtin_amdgcn_readfirstlane(x); };
    const int gy = static_cast<int>(opus::block_id_y()), gz = static_cast<int>(opus::block_id_z());
    const int by_group = k.causal ? 1 : 0;
    const int is_q = sc(gy >= (by_group ? k.n_kv_blocks : k.kv_rows) ? 1 : 0);
    const int nb   = is_q ? k.n_q_blocks : k.n_kv_blocks;
    // The coordinate that holds the group, over what it counts there (div_magic, div_magic_q).
    const int dv   = by_group ? gz : (is_q ? gy - k.kv_rows : gy);
    const int grp  = static_cast<int>((static_cast<unsigned long long>(static_cast<unsigned int>(dv)) *
                                       static_cast<unsigned int>(is_q ? k.div_magic_q : k.div_magic)) >> 31);
    const int rem  = dv - grp * (by_group ? k.num_seqs : nb);
    const int g    = sc(by_group ? rem : gz);
    // The sequence's bounds, seqstart[g] and [g + 1] of the queries and of the keys: two scalar
    // loads that go out here; a pass waits for them past its lanes' constants.
    unsigned long long bnd_q, bnd_k;
    {
        const int* pq = k.ptr_seqstart_q + g;
        const int* pk = k.ptr_seqstart_k + g;
        // Through the constant address space: scalar loads (the pins in front make a global load a vector one).
        auto u64 = [](const int* q) {
            // volatile: issued here and not moved into the passes; the wait is at the first use.
            const volatile auto* c = (const volatile __attribute__((address_space(4))) unsigned int*)(q);
            return c[0] | static_cast<unsigned long long>(c[1]) << 32;
        };
        bnd_q = u64(pq);
        bnd_k = u64(pk);
    }
    const int row  = by_group ? (is_q ? gy - k.n_kv_blocks : gy) : rem;
    const int hg   = fmha_bwd_d192_v128::wg_heads(k.causal);
    const int x    = static_cast<int>(opus::block_id_x());
    // A last group of `left` < hg heads: its ids, row by row, are its (block, head) pairs in order.
    const int left = k.H - grp * hg;
    const int lin  = row * (k.H < hg ? k.H : hg) + x;
    const int quo  = static_cast<int>((static_cast<unsigned long long>(static_cast<unsigned int>(lin)) * static_cast<unsigned int>(k.rem_magic)) >> 31);
    const int blk0 = left < hg ? quo : row;
    const int h    = sc(grp * hg + (left < hg ? lin - quo * left : x));
    // The head whose K and V head h reads.
    const int hk   = sc(static_cast<int>((static_cast<unsigned long long>(static_cast<unsigned int>(h)) * static_cast<unsigned int>(k.kv_magic)) >> 31));
    // Blocks 2 and 3 of every four change places: a zig-zag dispatch order.
    const int blk  = sc((blk0 & 2) && (blk0 | 1) < nb ? blk0 ^ 1 : blk0);
    // Ids past the pass's blocks: those a packed group leaves in its last row.
    if(blk0 >= nb)
    {
        __builtin_amdgcn_endpgm();
        __builtin_unreachable();
    }
    // Each pass ends the kernel itself: with the two ends joined, the compiler turns the if / else
    // into "if(!is_q) dK / dV; if(is_q) dQ" and keeps what the second needs (the arguments, the
    // zeroed accumulators) alive across the first, out of registers.
    if(is_q)
    {
        // A sequence's queries, m 64s of them, go in ceil(m / 4) blocks: of 256 (4 M tiles a wave),
        // but for the last -m mod 4 of them (at most all), of 192 (3 M tiles a wave), so that the
        // blocks hold fewer than 192 rows past the sequence's last. The long query blocks go first.
        constexpr int B4 = fmha_bwd_d192_v128_dq::NW * 64, B3 = fmha_bwd_d192_v128_dq::NW * 48;
        const int seqlen_q = static_cast<int>(bnd_q >> 32) - static_cast<int>(bnd_q);
        const int qb = k.n_q_blocks - 1 - blk;
        const int m = (seqlen_q + 63) >> 6, n = (m + 3) >> 2;
        const int n4 = n - ((-m & 3) < n ? (-m & 3) : n);
        if(qb < n4)
        {
            fmha_bwd_d192_v128_dq_body<4>(k, smem, lane, w, h, hk, sc(qb * B4), bnd_q, bnd_k, z);
            __builtin_amdgcn_endpgm();
            __builtin_unreachable();
        }
        fmha_bwd_d192_v128_dq_body<3>(k, smem, lane, w, h, hk, sc(n4 * B4 + (qb - n4) * B3), bnd_q, bnd_k, z);
        __builtin_amdgcn_endpgm();
        __builtin_unreachable();
    }
    else
    {
        fmha_bwd_d192_v128_dkdv_body(k, smem, lane, w, h, hk, blk, bnd_q, bnd_k, z);
        __builtin_amdgcn_endpgm();
        __builtin_unreachable();
    }
}
// clang-format on
