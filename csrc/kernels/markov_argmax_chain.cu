// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// markov_argmax_chain: K greedy steps of a first-order Markov chain whose transition logits carry a
// low-rank bias, in one persistent launch (gfx1250). With x_0 = ids[0], for k = 0 .. K-1:
//
//     rows_out[k] = W1[x_k]
//     x_{k+1}     = argmax_v ( float(base[k, v]) + sum_r W1[x_k, r] * W2[v, r] )      ids[k + 1] =
//     x_{k+1}
//
// (lowest id among equal maxima; NaN never wins; a step with no non-NaN logit answers id 0). A
// speculative block drafter samples its K positions this way, each step's argmax feeding the next
// step's W1 row.
//
// Layout. One workgroup per CU owns a contiguous run of 16-id vocab tiles. At the start its W2 rows
// are copied (TDM) into LDS rows padded by 16 B, up to `lds_tiles` tiles; each wave holds up to
// kRemMax more tiles in VGPRs, so the steps read no W2 from memory. Per tile and step: RANK / 32
// chained v_wmma_f32_16x16x32_bf16 (W1 row in the first operand, the W2 tile in the second, K split
// as lanes 0-15 = K 0-7|16-23 and lanes 16-31 = K 8-15|24-31 of each 32-chunk), then v =
// float(base) + acc.
//
// Exchange. After each step every workgroup publishes its (max, id) to a 64-bit slot tagged with
// this launch's tag, then every workgroup reduces all slots of the step itself. The tag is one past
// the one in the workgroup's own column of `slots`; each launch stamps its whole column (steps it
// does not run included), so a slot can only carry the current tag if this launch wrote it,
// whatever K the previous launches ran. The waits are bounded: a workgroup that never becomes
// resident (other work holding CUs) cannot hang the launch, it sets bit k of `err` for a step k
// whose wait gave up, and that step's ids are not to be trusted.

#include "aiter_tensor.h"

// The error bridge needs aiter_hip_common.h, which aiter_tensor.h includes.
#include "aiter_ctypes_error.h"
#include <hip/hip_runtime.h>
#include <mutex>
#include <stdint.h>
#include <string>
#include <unordered_map>

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1250__) || defined(__gfx1250_strict__))
#include "opus/opus.hpp"
#endif

namespace aiter {
namespace markov_argmax_chain {

constexpr int kThreads   = 256;
constexpr int kWaves     = 8;
constexpr int kMaxSteps  = 32;     // one bit of `err` a step
constexpr int kPollBound = 400000; // s_sleep 16 polls before a wait gives up
constexpr int kRemMax    = 2;      // tiles past LDS a wave holds in VGPRs
constexpr int kTpwMax    = 8; // tiles a wave handles in total (LDS + VGPR), the base prefetch bound
constexpr int kIdBits    = 18;     // id field of a slot; 0x3ffff is the "no id" sentinel
constexpr int kTagMask   = 0x3fff; // 14-bit launch tag
constexpr int kScratch   = 512;    // LDS past the tiles: the per-wave reduction
constexpr int kLdsBytes  = 327680; // gfx1250 LDS a workgroup can hold
constexpr int kMaxLdsTile = kWaves * (kTpwMax - kRemMax);

template <int RANK>
struct Geometry
{
    static_assert(RANK % 32 == 0 && RANK >= 64 && RANK <= 256, "RANK: 64..256, a multiple of 32");
    static constexpr int kChunks = RANK / 32; // WMMAs a tile
    static constexpr int kRowPitch =
        RANK * 2 + 16; // one LDS row + 16 B TDM pad: rows of a tile hit distinct banks
    static constexpr int kTileLds  = 16 * kRowPitch;
    static constexpr int kLdsTiles = (kLdsBytes - kScratch) / kTileLds < kMaxLdsTile
                                         ? (kLdsBytes - kScratch) / kTileLds
                                         : kMaxLdsTile;
};

#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1250__) || defined(__gfx1250_strict__))

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

struct U32x8
{
    u32x4 lo, hi;
};
__device__ __forceinline__ bf16x16 make_op(u32x4 lo, u32x4 hi)
{
    U32x8 t{lo, hi};
    return __builtin_bit_cast(bf16x16, t);
}
__device__ __forceinline__ int koff(int j, int h, int part) { return 32 * j + 8 * h + 16 * part; }
// D[0][n] = sum_k w1row[k] * W2[n][k]: output id n of the tile in lane n (n < 16), element 0
__device__ __forceinline__ f32x8 wmma_step(bf16x16 w2op, bf16x16 eop, f32x8 acc)
{
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        false, eop, false, w2op, (short)0, acc, false, false);
}
__device__ __forceinline__ bool better(float v, int i, float bv, int bi)
{
    return v > bv || (v == bv && i < bi);
}
__device__ __forceinline__ float bf16_lo(unsigned int w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ void wave_argmax(float& v, int& i)
{
#pragma unroll
    for(int m = 16; m >= 1; m >>= 1)
    {
        float ov = __shfl_xor(v, m, 32);
        int oi   = __shfl_xor(i, m, 32);
        if(better(ov, oi, v, i))
        {
            v = ov;
            i = oi;
        }
    }
}
__device__ __forceinline__ unsigned long long ld_dev(const unsigned long long* p)
{
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void st_dev(unsigned long long* p, unsigned long long v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ unsigned long long pack_slot(float v, int id, unsigned tag)
{
    return ((unsigned long long)__float_as_uint(v) << 32) |
           ((unsigned long long)(id & ((1 << kIdBits) - 1)) << 14) | (tag & kTagMask);
}

// one LDS tile's WMMA chain (row r of the tile at tb): all LDS reads first, then the chain
template <int RANK>
__device__ __forceinline__ f32x8 tile_lds(const unsigned char* tb, int h, const bf16x16* eb)
{
    constexpr int NC = Geometry<RANK>::kChunks;
    f32x8 acc        = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    u32x4 lo[NC], hi[NC];
#pragma unroll
    for(int j = 0; j < NC; ++j)
    {
        lo[j] = *(const u32x4*)(tb + ((koff(j, h, 0) >> 3) << 4));
        hi[j] = *(const u32x4*)(tb + ((koff(j, h, 1) >> 3) << 4));
    }
#pragma unroll
    for(int j = 0; j < NC; ++j)
        acc = wmma_step(make_op(lo[j], hi[j]), eb[j], acc);
    return acc;
}

template <int RANK, typename IdT>
__global__ void __launch_bounds__(kThreads, 1)
    markov_argmax_chain_kernel(const __bf16* __restrict__ base,
                               long base_stride,
                               IdT* __restrict__ ids,
                               long ids_stride,
                               const __bf16* __restrict__ w1,
                               const __bf16* __restrict__ w2,
                               __bf16* __restrict__ rows_out,
                               long rows_stride,
                               unsigned long long* __restrict__ slots,
                               int slot_rows,
                               int* __restrict__ err,
                               int V,
                               int nsteps,
                               int lds_tiles)
{
    using G                 = Geometry<RANK>;
    constexpr int NC        = G::kChunks;
    constexpr int ROW_PITCH = G::kRowPitch;
    constexpr int TILE_LDS  = G::kTileLds;

    extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int ncta = gridDim.x, cta = blockIdx.x;
    const int h = lane >> 4, r = lane & 15;
    const bool live = (r == 0);

    const int ntiles = V >> 4;
    const int q = ntiles / ncta, rem = ntiles % ncta;
    const int t0 = cta * q + min(cta, rem);
    const int nt = q + (cta < rem ? 1 : 0);
    const int nl = min(nt, lds_tiles);

    unsigned char* scratch = lds + (size_t)lds_tiles * TILE_LDS;
    float* red_v           = (float*)scratch;
    int* red_i             = (int*)(red_v + kWaves);
    int* bcast             = red_i + kWaves;

    // This launch's tag: one past the last launch's, read from this workgroup's own column (only
    // this workgroup writes it, and every launch stamps all of it).
    const unsigned tag = ((unsigned)(ld_dev(&slots[cta]) & kTagMask) + 1) & kTagMask;
    // The rows of steps this launch does not run carry its tag too, so that no later launch can
    // find one of them still holding the tag it is about to use, however many launches with fewer
    // steps come in between.
    if(tid >= nsteps && tid < slot_rows)
        st_dev(&slots[(size_t)tid * ncta + cta],
               pack_slot(-__builtin_inff(), (1 << kIdBits) - 1, tag));

    // ---- prologue 1: LDS fill. This workgroup's nl tiles are nl * 16 consecutive W2 rows; wave w
    // loads rows [w * rpw, ...) in ops of 32 rows, at most 3 in flight.
    {
        using Win       = opus::tdm<unsigned short,
                                    opus::seq<RANK, 32>,
                                    opus::tdm_traits::padding<unsigned short, RANK, 8>>;
        const int nrows = nl * 16;
        const int rpw   = (nrows + kWaves - 1) / kWaves;
        const int r0 = min(nrows, wave * rpw), r1 = min(nrows, r0 + rpw);
        if(r1 > r0)
        {
            const unsigned lds_base = (unsigned)reinterpret_cast<__UINTPTR_TYPE__>(lds);
            Win win                 = opus::make_tdm<Win>(lds_base,
                                          w2 + (size_t)t0 * 16 * RANK,
                                          (opus::u32_t)RANK,
                                          (opus::u32_t)r1,
                                          (opus::u64_t)RANK,
                                          0u,
                                          (opus::u32_t)r0);
            int issued              = 0;
            for(int rr = r0; rr < r1; rr += 32)
            {
                if(issued == 3)
                {
                    __builtin_amdgcn_s_wait_tensorcnt(2);
                    --issued;
                }
                // exactly this wave's rows: a tile past the tensor extent would be zero-filled into
                // the next wave's rows
                win.set_tile_dim1((opus::u32_t)min(32, r1 - rr));
                win.async_load((opus::u32_t)(rr * (ROW_PITCH / 2)));
                win.move(opus::number<0>{}, 32);
                ++issued;
            }
            __builtin_amdgcn_s_wait_tensorcnt(0);
        }
    }
    // ---- prologue 2: this wave's tiles past LDS (nl + wave + 8 i) into VGPRs, in operand layout
    // ----
    u32x4 rlo[kRemMax][NC], rhi[kRemMax][NC];
#pragma unroll
    for(int i = 0; i < kRemMax; ++i)
    {
        const int t    = min(nl + wave + kWaves * i,
                          nt - 1); // branch-free; a clamped duplicate is never consumed
        const u32x4* g = (const u32x4*)(w2 + ((size_t)(t0 + t) * 16 + r) * RANK);
#pragma unroll
        for(int j = 0; j < NC; ++j)
        {
            rlo[i][j] = __builtin_nontemporal_load(g + (koff(j, h, 0) >> 3));
            rhi[i][j] = __builtin_nontemporal_load(g + (koff(j, h, 1) >> 3));
        }
    }
    // The tiles this wave handles: LDS tiles wave, wave + 8, ... (< nl), then its VGPR tiles. Lane
    // r's base value of each, for one step, prefetched into registers.
    unsigned bpre[kTpwMax];
    auto load_base = [&](int k) {
        const __bf16* bk = base + (size_t)k * base_stride;
#pragma unroll
        for(int s = 0; s < kTpwMax; ++s)
        {
            int t   = (s < kTpwMax - kRemMax) ? (wave + kWaves * s)
                                              : (nl + wave + kWaves * (s - (kTpwMax - kRemMax)));
            t       = min(t, nt - 1); // branch-free; a clamped slot is never consumed
            bpre[s] = *(const unsigned short*)(bk + (t0 + t) * 16 + r);
        }
    };
    load_base(0);
    __syncthreads();

    int prev = (int)ids[0];
    for(int k = 0; k < nsteps; ++k)
    {
        const __bf16* e = w1 + (size_t)prev * RANK;
        bf16x16 eb[NC];
#pragma unroll
        for(int j = 0; j < NC; ++j)
        {
            // every lane loads (one row), lanes past column 0 zeroed by select
            u32x4 lo      = *(const u32x4*)(e + koff(j, h, 0));
            u32x4 hi      = *(const u32x4*)(e + koff(j, h, 1));
            const u32x4 z = {0, 0, 0, 0};
            eb[j]         = make_op(live ? lo : z, live ? hi : z);
        }
        if(cta == 0 && tid < RANK / 8)
            *(u32x4*)(rows_out + (size_t)k * rows_stride + tid * 8) = *(const u32x4*)(e + tid * 8);

        float bv     = -__builtin_inff();
        int bi       = 0x7fffffff;
        auto consume = [&](const f32x8& acc, int t, unsigned b) {
            if(h == 0)
            {
                const int vid = (t0 + t) * 16 + r;
                float v       = bf16_lo(b) + acc[0];
                if(better(v, vid, bv, bi))
                {
                    bv = v;
                    bi = vid;
                }
            }
        };
        // LDS tiles, two a loop iteration (two independent WMMA chains)
#pragma unroll
        for(int s = 0; s < kTpwMax - kRemMax; s += 2)
        {
            const int ta = wave + kWaves * s, tb = wave + kWaves * (s + 1);
            if(ta < nl)
            {
                f32x8 aa = tile_lds<RANK>(lds + (size_t)ta * TILE_LDS + r * ROW_PITCH, h, eb);
                if(tb < nl)
                {
                    f32x8 ab = tile_lds<RANK>(lds + (size_t)tb * TILE_LDS + r * ROW_PITCH, h, eb);
                    consume(aa, ta, bpre[s]);
                    consume(ab, tb, bpre[s + 1]);
                }
                else
                {
                    consume(aa, ta, bpre[s]);
                }
            }
        }
        // VGPR tiles
#pragma unroll
        for(int i = 0; i < kRemMax; ++i)
        {
            const int t = nl + wave + kWaves * i;
            if(t < nt)
            {
                f32x8 acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for(int j = 0; j < NC; ++j)
                    acc = wmma_step(make_op(rlo[i][j], rhi[i][j]), eb[j], acc);
                consume(acc, t, bpre[kTpwMax - kRemMax + i]);
            }
        }
        // the next step's base values: issued now, consumed after the exchange
        if(k + 1 < nsteps)
            load_base(k + 1);

        // this workgroup's (max, id) into its slot of step k
        wave_argmax(bv, bi);
        if(lane == 0)
        {
            red_v[wave] = bv;
            red_i[wave] = bi;
        }
        __syncthreads();
        if(tid == 0)
        {
            float cv = red_v[0];
            int ci   = red_i[0];
            for(int w = 1; w < kWaves; ++w)
                if(better(red_v[w], red_i[w], cv, ci))
                {
                    cv = red_v[w];
                    ci = red_i[w];
                }
            if(ci == 0x7fffffff)
                ci = (1 << kIdBits) - 1;
            st_dev(&slots[(size_t)k * ncta + cta], pack_slot(cv + 0.0f, ci, tag));
        }

        // every workgroup reduces all slots of step k: thread j waits for workgroup j's
        float pv = -__builtin_inff();
        int pi   = 0x7fffffff;
        if(tid < ncta)
        {
            unsigned long long sv;
            int it = 0;
            for(;;)
            {
                sv = ld_dev(&slots[(size_t)k * ncta + tid]);
                if((unsigned)(sv & kTagMask) == tag)
                    break;
                if(++it > kPollBound)
                {
                    atomicOr(err, (int)(1u << k));
                    break;
                }
                __builtin_amdgcn_s_sleep(16);
            }
            pv = __uint_as_float((unsigned)(sv >> 32));
            pi = (int)((sv >> 14) & ((1 << kIdBits) - 1));
        }
        wave_argmax(pv, pi);
        __syncthreads();
        if(lane == 0)
        {
            red_v[wave] = pv;
            red_i[wave] = pi;
        }
        __syncthreads();
        if(tid == 0)
        {
            float cv = red_v[0];
            int ci   = red_i[0];
            for(int w = 1; w < kWaves; ++w)
                if(better(red_v[w], red_i[w], cv, ci))
                {
                    cv = red_v[w];
                    ci = red_i[w];
                }
            if(ci >= V)
                ci = 0; // no non-NaN logit in the step: id 0
            bcast[0] = ci;
        }
        __syncthreads();
        const int winner = bcast[0];
        if(cta == 0 && tid == 0)
            ids[(size_t)(k + 1) * ids_stride] = (IdT)winner;
        prev = winner;
        __syncthreads();
    }
}

#else // every other device pass, and the host pass: the kernel's symbol only

template <int RANK, typename IdT>
__global__ void __launch_bounds__(kThreads, 1)
    markov_argmax_chain_kernel(const __bf16* __restrict__,
                               long,
                               IdT* __restrict__,
                               long,
                               const __bf16* __restrict__,
                               const __bf16* __restrict__,
                               __bf16* __restrict__,
                               long,
                               unsigned long long* __restrict__,
                               int,
                               int* __restrict__,
                               int,
                               int,
                               int)
{
}

#endif

// Tiles a workgroup holds at most: up to lds_tiles in LDS, kRemMax a wave in VGPRs.
template <int RANK>
inline bool tile_split_fits(int V, int ncta)
{
    const int most = (V >> 4) / ncta + 1;
    return most <= Geometry<RANK>::kLdsTiles + kWaves * kRemMax;
}

struct DeviceInfo
{
    bool gfx1250;
    int num_cus;
};

inline DeviceInfo device_info(int device)
{
    static std::mutex mu;
    static std::unordered_map<int, DeviceInfo> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(device);
    if(it != cache.end())
        return it->second;
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, device));
    std::string arch = prop.gcnArchName;
    arch             = arch.substr(0, arch.find(':'));
    DeviceInfo info{arch == "gfx1250" || arch == "gfx1250-strict", prop.multiProcessorCount};
    cache.emplace(device, info);
    return info;
}

template <int RANK, typename IdT>
void launch(const aiter_tensor_t* base,
            const aiter_tensor_t* ids,
            const aiter_tensor_t* w1,
            const aiter_tensor_t* w2,
            const aiter_tensor_t* rows_out,
            const aiter_tensor_t* slots,
            const aiter_tensor_t* err,
            hipStream_t stream)
{
    using G        = Geometry<RANK>;
    const int V    = (int)base->size(1);
    const int ncta = (int)slots->size(1);
    AITER_CHECK(tile_split_fits<RANK>(V, ncta),
                "markov_argmax_chain: vocab ",
                V,
                " does not fit ",
                ncta,
                " workgroups at rank ",
                RANK,
                " (at most ",
                G::kLdsTiles + kWaves * kRemMax,
                " 16-id tiles a workgroup)");
    auto kern     = markov_argmax_chain_kernel<RANK, IdT>;
    const int lds = G::kLdsTiles * G::kTileLds + kScratch;
    HIP_CALL(
        hipFuncSetAttribute((const void*)kern, hipFuncAttributeMaxDynamicSharedMemorySize, lds));
    hipLaunchKernelGGL(kern,
                       dim3(ncta),
                       dim3(kThreads),
                       lds,
                       stream,
                       (const __bf16*)base->data_ptr(),
                       (long)base->stride(0),
                       (IdT*)ids->data_ptr(),
                       (long)ids->stride(0),
                       (const __bf16*)w1->data_ptr(),
                       (const __bf16*)w2->data_ptr(),
                       (__bf16*)rows_out->data_ptr(),
                       (long)rows_out->stride(0),
                       (unsigned long long*)slots->data_ptr(),
                       (int)slots->size(0),
                       (int*)err->data_ptr(),
                       V,
                       (int)base->size(0),
                       G::kLdsTiles);
    HIP_CALL(hipGetLastError());
}

inline void check_gpu(const aiter_tensor_t* t, const char* name, int device)
{
    AITER_CHECK(t != nullptr, "markov_argmax_chain: ", name, " must not be null");
    AITER_CHECK(t->is_gpu() && t->device_id == device,
                "markov_argmax_chain: ",
                name,
                " must be on the device of base_logits");
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

} // namespace markov_argmax_chain
} // namespace aiter

AITER_CTYPES_ERROR_DEF

// base_logits [K, V] bf16 (unit stride along V); ids [K + 1] int32 / int64, any stride (ids[0] is
// read, ids[1:] written); w1, w2 [V, R] bf16 contiguous, R 128 or 256; rows_out [K, R] bf16 (unit
// stride along R, 16-B aligned rows); slots [S, C] int64 contiguous, S >= K, C the grid (<= the CU
// count, <= 256); err [1] int32.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(markov_argmax_chain_fwd,
                                    (aiter_tensor_t * base_logits,
                                     aiter_tensor_t* ids,
                                     aiter_tensor_t* w1,
                                     aiter_tensor_t* w2,
                                     aiter_tensor_t* rows_out,
                                     aiter_tensor_t* slots,
                                     aiter_tensor_t* err,
                                     hipStream_t stream),
                                    (base_logits, ids, w1, w2, rows_out, slots, err, stream))
{
    using namespace aiter::markov_argmax_chain;
    AITER_CHECK(base_logits != nullptr && base_logits->is_gpu(),
                "markov_argmax_chain: base_logits must be a GPU tensor");
    const int device = base_logits->device_id;
    check_gpu(ids, "ids", device);
    check_gpu(w1, "w1", device);
    check_gpu(w2, "w2", device);
    check_gpu(rows_out, "rows_out", device);
    check_gpu(slots, "slots", device);
    check_gpu(err, "err", device);
    const DeviceInfo dev = device_info(device);
    AITER_CHECK(dev.gfx1250, "markov_argmax_chain: gfx1250 only");

    AITER_CHECK(base_logits->dim() == 2 && base_logits->dtype() == AITER_DTYPE_bf16 &&
                    base_logits->stride(1) == 1,
                "markov_argmax_chain: base_logits must be bf16 [K, V] with unit stride along V");
    const int64_t K = base_logits->size(0), V = base_logits->size(1);
    AITER_CHECK(K >= 1 && K <= kMaxSteps,
                "markov_argmax_chain: K = ",
                K,
                " steps, supported 1..",
                kMaxSteps);
    AITER_CHECK(V >= 16 && V % 16 == 0 && V < (1 << kIdBits),
                "markov_argmax_chain: vocab ",
                V,
                " must be a multiple of 16 below ",
                1 << kIdBits);

    AITER_CHECK(w1->dim() == 2 && w2->dim() == 2 && w1->dtype() == AITER_DTYPE_bf16 &&
                    w2->dtype() == AITER_DTYPE_bf16 && w1->is_contiguous() && w2->is_contiguous(),
                "markov_argmax_chain: w1 and w2 must be contiguous bf16 [V, R]");
    const int64_t R = w2->size(1);
    AITER_CHECK(w1->size(0) == V && w2->size(0) == V && w1->size(1) == R,
                "markov_argmax_chain: w1 and w2 must both be [V, R] with V = ",
                V);
    AITER_CHECK(R == 128 || R == 256, "markov_argmax_chain: rank ", R, " is not built (128, 256)");
    AITER_CHECK(aligned16(w1->data_ptr()) && aligned16(w2->data_ptr()),
                "markov_argmax_chain: w1 and w2 must be 16-byte aligned");

    AITER_CHECK(ids->dim() == 1 && ids->size(0) == K + 1 &&
                    (ids->dtype() == AITER_DTYPE_i32 || ids->dtype() == AITER_DTYPE_i64),
                "markov_argmax_chain: ids must be int32 or int64 [K + 1]");
    AITER_CHECK(
        rows_out->dim() == 2 && rows_out->dtype() == AITER_DTYPE_bf16 && rows_out->size(0) == K &&
            rows_out->size(1) == R && rows_out->stride(1) == 1 && rows_out->stride(0) % 8 == 0 &&
            aligned16(rows_out->data_ptr()),
        "markov_argmax_chain: rows_out must be bf16 [K, R] with unit stride along R and 16-byte "
        "aligned rows");
    AITER_CHECK(
        slots->dim() == 2 && slots->dtype() == AITER_DTYPE_i64 && slots->is_contiguous() &&
            slots->size(0) >= K && slots->size(0) <= kMaxSteps,
        "markov_argmax_chain: slots must be a contiguous int64 [S, C] workspace with K <= S <= ",
        kMaxSteps);
    AITER_CHECK(slots->size(1) >= 1 && slots->size(1) <= dev.num_cus && slots->size(1) <= kThreads,
                "markov_argmax_chain: slots has ",
                slots->size(1),
                " columns (the grid); it needs 1 to min(",
                dev.num_cus,
                " CUs, ",
                kThreads,
                ") so that every workgroup is resident");
    AITER_CHECK(err->numel() >= 1 && err->dtype() == AITER_DTYPE_i32,
                "markov_argmax_chain: err must be int32 [1]");

    HipDeviceGuard guard(device);
    const bool id64 = ids->dtype() == AITER_DTYPE_i64;
    if(R == 256)
    {
        if(id64)
            launch<256, long long>(base_logits, ids, w1, w2, rows_out, slots, err, stream);
        else
            launch<256, int>(base_logits, ids, w1, w2, rows_out, slots, err, stream);
    }
    else
    {
        if(id64)
            launch<128, long long>(base_logits, ids, w1, w2, rows_out, slots, err, stream);
        else
            launch<128, int>(base_logits, ids, w1, w2, rows_out, slots, err, stream);
    }
}
