// SPDX-License-Identifier: MIT
#include "params.cuh"
#include "partial_io.cuh"
#include "q_encode.cuh"
#define PV_PIPE_WAIT 1
namespace sparse_mla_fp8::small {

#ifndef CACHEX_READ_NT
#define CACHEX_READ_NT 0
#endif
#ifndef CACHEX_STORE_NT
#define CACHEX_STORE_NT 0
#endif
#ifndef LOAD_MODE
#define LOAD_MODE 0
#endif
#ifndef IDS_MODE
#define IDS_MODE 0
#endif
// SPDX-License-Identifier: MIT
using I8              = int __attribute__((ext_vector_type(8)));
using U4              = uint32_t __attribute__((ext_vector_type(4)));
using F4              = float __attribute__((ext_vector_type(4)));
using Params          = sparse_mla_fp8::Params;
constexpr float LOG2E = 1.4426950408889634f, LN2 = .6931471805599453f;
__device__ __forceinline__ F4 mma(I8 a, I8 b, F4 c)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}
// unchanged. Keep producer/consumer hazards inside the same assembly block:
// Table38 non-DL VALU->MFMA 2 waits; XDL result->VALU 12 waits for this form.
template <bool VGPR>
__device__ __forceinline__ F4 online_pv_mma(I8 a, I8 b, F4 c)
{
    if constexpr(!VGPR)
        return mma(a, b, c);
    else
    {
        const int scale = 127;
        asm volatile(
            "s_nop 1\n\t"
            "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %0, %3, %3 op_sel_hi:[0,0,0]\n\t"
            "s_nop 7\n\ts_nop 3"
            : "+&v"(c)
            : "v"(a), "v"(b), "v"(scale)
            : "memory");
        return c;
    }
}
__device__ __forceinline__ uint32_t pack4(F4 x)
{
#pragma unroll
    for(int j = 0; j < 4; ++j)
        x[j] = fminf(448.f, fmaxf(-448.f, x[j]));
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], 0, 0);
    return __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], w, 1);
}
// Each 1 KiB chunk is 16 rows x 64 bytes. XOR preserves 16-byte vectors.
#include "kv_layout.cuh"
template <int Rows>
__device__ __forceinline__ I8 fragment(const uint8_t* base, int row, int col)
{
    I8 x;
    const U4 a = *reinterpret_cast<const U4*>(base + address<Rows>(row, col));
    const U4 b = *reinterpret_cast<const U4*>(base + address<Rows>(row, col + 16));
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
__device__ __forceinline__ int p_address(int head, int key)
{
    return head * 128 + (key ^ ((head & 7) * 16));
}
__device__ __forceinline__ I8 p_fragment(const uint8_t* base, int head, int col)
{
    I8 x;
    U4 a = *reinterpret_cast<const U4*>(base + p_address(head, col));
    U4 b = *reinterpret_cast<const U4*>(base + p_address(head, col + 16));
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[j]     = a[j];
        x[j + 4] = b[j];
    }
    return x;
}
__device__ __forceinline__ I8 v_fragment(const uint8_t* base, int col, int key)
{
    I8 x{};
#pragma unroll
    for(int j = 0; j < 8; ++j)
    {
        uint32_t w = 0;
#pragma unroll
        for(int b = 0; b < 4; ++b)
            w |= uint32_t(base[address<128>(key + j * 4 + b, col)]) << (8 * b);
        x[j] = w;
    }
    return x;
}
using I2 = int __attribute__((ext_vector_type(2)));
__device__ __forceinline__ I8 v_fragment_tr(const uint8_t* base, int col_base)
{
    const int lane = threadIdx.x & 63;
    I8 x;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int row       = (lane / 16) * 32 + i * 8 + (lane % 16) / 2;
        int col       = col_base + (lane % 2) * 8;
        uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(base + address<128>(row, col)));
        I2 r;
        asm volatile("ds_read_b64_tr_b8 %0, %1" : "=v"(r) : "v"(addr) : "memory");
        x[i * 2]     = r[0];
        x[i * 2 + 1] = r[1];
    }
    // Tie the loaded operand to the wait. A memory-only barrier can move
    // after the last register-only MFMA when its results stay in registers.
    asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : "+v"(x) : : "memory");
    return x;
}
__device__ __forceinline__ void direct_load(uint8_t* dest, const uint8_t* src)
{
    uint32_t lds  = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(dest)));
    uint64_t addr = reinterpret_cast<uintptr_t>(src);
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, off"
                 :
                 : "s"(lds), "v"(addr)
                 : "m0", "memory");
}
__host__ __device__ constexpr bool narrow_kv(int slots)
{
    return slots >= 0 && uint64_t(slots) * 512 <= 0x7fffffffULL;
}
template <bool Narrow>
__device__ __forceinline__ void
direct_kv(uint8_t* dest, const uint8_t* base, int id, int col, int slots)
{
    if constexpr(Narrow && LOAD_MODE)
    {
        uint32_t lds  = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(dest)));
        uint64_t addr = reinterpret_cast<uintptr_t>(base);
        uint32_t offset = (uint32_t(id) << 9) + uint32_t(col);
#if LOAD_MODE == 1
        U4 resource = {uint32_t(addr), uint32_t(addr >> 32), uint32_t(slots) * 512, 0x00020000};
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dwordx4 %1, %2, 0 offen lds"
                     :
                     : "s"(lds), "v"(offset), "s"(resource)
                     : "m0", "memory");
#elif LOAD_MODE == 2
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tglobal_load_lds_dwordx4 %1, %2"
                     :
                     : "s"(lds), "v"(offset), "s"(addr)
                     : "m0", "memory");
#endif
    }
    else
        direct_load(dest, base + int64_t(id) * 512 + col);
}

#include "partial_layout.cuh"
#include "pv_pipe.cuh"
#include "softmax_lane.cuh"
// Pad16 addresses keep the byte payload and arithmetic unchanged. Offsets are
// compile-time instruction immediates; no array of32 persistent PV addresses.
#include "pad16_factored.cuh"

template <int Splits,
          int Order,
          bool Prefetch,
          bool TailBarrier,
          int Waves        = 8,
          bool Narrow      = false,
          bool QBF16       = false,
          bool QUnitScale  = false,
          int PartialTile  = 0,
          bool Remap       = false,
          bool NT          = false,
          int PVGroup      = 1,
          int IdsMode      = 0,
          bool PadFactored = false,
          bool Online      = false,
          bool PVVGPR      = false>
__global__ __launch_bounds__(Waves * 64) void attention(Params p)
{
    constexpr int Tiles = (17 + Splits - 1) / Splits;
    constexpr int Pitch = PadFactored ? 1040 : KV_STRIDE;
    static_assert(!PadFactored || (PVGroup == 1 && IdsMode == 0));
    constexpr bool Transpose = true, Direct = true, LocalReduce = true, LoadSchedule = true;
    __shared__ __align__(16) uint8_t kvs[64 * Pitch];
    __shared__ __align__(16) uint8_t qs[8192];
    __shared__ __align__(16) uint8_t ps[2048];
    __shared__ float exchange[64], row_max[16], row_inv[16];
    __shared__ float exchange_sum[LocalReduce ? 64 : 1];
    __shared__ int selected[IdsMode == 0 ? 128 : 1];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid / 64, head = lane & 15,
              group = lane / 16;
    static_assert(Order == 0 && (PVGroup == 1 || PVGroup == 4));
    CtaTile work;
    if constexpr(Remap)
    {
        const auto t = cta_tile<4, Splits, true>();
        work         = {t.q, t.c, t.b};
    }
    else
        work = {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z)};
    const int query = work.q, part = work.b, hbase = work.c * 16;
    const int start = (query * p.capacity), end = ((query + 1) * p.capacity);
    // Uniform within the CTA. Distribute the seventeenth tail tile without
    // computing ceil(17/S) tiles for every split. No host reads of selection values.
    const int total_tiles = (end - start + 127) / 128;
    const int quotient = total_tiles / Splits, remainder = total_tiles % Splits;
    const int tile_begin = part * quotient + (part < remainder ? part : remainder);
    const int tile_count = quotient + (part < remainder ? 1 : 0);
    if(wave < 4)
    {
#pragma unroll
        for(int i = 0; i < 2; ++i)
        {
            int chunk = wave + i * 4;
            int r = lane / 4, c = chunk * 64 + ((lane % 4) * 16 ^ ((r & 3) * 16));
            const int64_t off =
                int64_t(query) * p.q_stride_row + int64_t(hbase + r) * p.q_stride_head + c;
            if constexpr(QBF16)
            {
                const auto* src = reinterpret_cast<const uint16_t*>(p.q) + off;
                U4 encoded      = q_encode::pack16<QUnitScale>(src, p.q_scale);
                *reinterpret_cast<U4*>(qs + chunk * 1024 + lane * 16) = encoded;
            }
            else if constexpr(Direct)
                direct_load(qs + chunk * 1024, p.q + off);
            else
                *reinterpret_cast<U4*>(qs + chunk * 1024 + lane * 16) =
                    *reinterpret_cast<const U4*>(p.q + off);
        }
    }
    F4 merged[8]{};
    float merge_max = -INFINITY, merge_sum = 0.f;
    int next_id = -1, next_mask_id = -1, next_id0 = -1, next_id1 = -1;
// Q is staged once per grouped CTA. KV/P shared storage is reused per tile.
#pragma unroll 1
    for(int iteration = 0; iteration < tile_count; ++iteration)
    {
        const int split     = tile_begin + iteration;
        uint32_t valid_bits = 0;
        int reg_id0 = -1, reg_id1 = -1;
        if constexpr(IdsMode == 0)
        {
            if(tid < 128)
            {
                int id;
                if constexpr(Prefetch)
                {
                    if(iteration == 0)
                    {
                        int at = start + split * 128 + tid;
                        id     = at < end ? p.indices[at] : -1;
                    }
                    else
                        id = next_id;
                }
                else
                {
                    int at = start + split * 128 + tid;
                    id     = at < end ? p.indices[at] : -1;
                }
                selected[tid] = (id >= 0 && id < p.slots) ? id : -1;
            }
            // Also protects the previous tile's KV/P consumers when TailBarrier=false.
            __syncthreads();
            if constexpr(Prefetch)
            {
                if(tid < 128 && iteration + 1 < tile_count)
                {
                    int at  = start + (split + 1) * 128 + tid;
                    next_id = at < end ? p.indices[at] : -1;
                }
            }
        }
        else
        {
            // First tile needs no ID exchange. Later tiles MUST wait for all
            // preceding KV/P consumers before overwriting shared storage.
            if constexpr(IdsMode == 3)
                __syncthreads();
            else if(iteration > 0 && !TailBarrier)
                __syncthreads();
            if(wave < 4)
            {
                int at = start + split * 128 + wave * 32 + (lane & 31);
                int id;
                if constexpr(IdsMode == 2 && Prefetch)
                    id = iteration ? next_mask_id : (at < end ? p.indices[at] : -1);
                else
                    id = at < end ? p.indices[at] : -1;
                valid_bits = uint32_t(__ballot(id >= 0 && id < p.slots));
            }
            if(Waves == 4 || wave >= 4)
            {
                int loader = Waves == 4 ? wave : wave - 4;
                int at0 = start + split * 128 + loader * 16 + lane / 4, at1 = at0 + 64;
                if constexpr(IdsMode == 2 && Prefetch)
                {
                    reg_id0 = iteration ? next_id0 : (at0 < end ? p.indices[at0] : -1);
                    reg_id1 = iteration ? next_id1 : (at1 < end ? p.indices[at1] : -1);
                }
                else
                {
                    reg_id0 = at0 < end ? p.indices[at0] : -1;
                    reg_id1 = at1 < end ? p.indices[at1] : -1;
                }
                reg_id0 = (reg_id0 >= 0 && reg_id0 < p.slots) ? reg_id0 : -1;
                reg_id1 = (reg_id1 >= 0 && reg_id1 < p.slots) ? reg_id1 : -1;
            }
            if constexpr(IdsMode == 2 && Prefetch)
            {
                if(iteration + 1 < tile_count)
                {
                    if(wave < 4)
                    {
                        int at       = start + (split + 1) * 128 + wave * 32 + (lane & 31);
                        next_mask_id = at < end ? p.indices[at] : -1;
                    }
                    if(Waves == 4 || wave >= 4)
                    {
                        int loader = Waves == 4 ? wave : wave - 4;
                        int at0    = start + (split + 1) * 128 + loader * 16 + lane / 4,
                            at1    = at0 + 64;
                        next_id0   = at0 < end ? p.indices[at0] : -1;
                        next_id1   = at1 < end ? p.indices[at1] : -1;
                    }
                }
            }
        }
        if(Waves == 4 || wave >= 4)
        {
            int loader = Waves == 4 ? wave : wave - 4;
            if constexpr(LoadSchedule)
            {
                int id0, id1;
                if constexpr(IdsMode == 0)
                {
                    id0 = selected[loader * 16 + lane / 4];
                    id1 = selected[(loader + 4) * 16 + lane / 4];
                }
                else
                {
                    id0 = reg_id0;
                    id1 = reg_id1;
                }
#pragma unroll
                for(int i = 0; i < 16; ++i)
                {
                    int chunk = loader + i * 4, r = (chunk % 8) * 16 + lane / 4;
                    int c  = (chunk / 8) * 64 + ((lane % 4) * 16 ^ kv_xor(r));
                    int id = (i & 1) ? id1 : id0;
                    if(id >= 0)
                        direct_kv<Narrow>(kvs + chunk * Pitch, p.kv, id, c, p.slots);
                    else
                        *reinterpret_cast<U4*>(kvs + chunk * Pitch + lane * 16) = U4{};
                }
            }
            else
            {
#pragma unroll
                for(int chunk = loader; chunk < 64; chunk += 4)
                {
                    int r = (chunk % 8) * 16 + lane / 4,
                        c = (chunk / 8) * 64 + ((lane % 4) * 16 ^ kv_xor(r));
                    int id;
                    if constexpr(IdsMode == 0)
                        id = selected[r];
                    else
                    {
                        int at = start + split * 128 + r;
                        id     = at < end ? p.indices[at] : -1;
                        id     = (id >= 0 && id < p.slots) ? id : -1;
                    }
                    if constexpr(Direct)
                    {
                        if(id >= 0)
                            direct_kv<Narrow>(kvs + chunk * Pitch, p.kv, id, c, p.slots);
                        else
                            *reinterpret_cast<U4*>(kvs + chunk * Pitch + lane * 16) = U4{};
                    }
                    else
                    {
                        U4 x{};
                        if(id >= 0)
                            x = *reinterpret_cast<const U4*>(p.kv + int64_t(id) * 512 + c);
                        *reinterpret_cast<U4*>(kvs + chunk * Pitch + lane * 16) = x;
                    }
                }
            }
        }
        if constexpr(Direct)
            asm volatile("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)" : : : "memory");
        __syncthreads();
        F4 a{}, b{};
        if(wave < 4)
        {
            if constexpr(PadFactored)
            {
                padded_qk_step<0>(qs, kvs, head, group, wave, a, b);
            }
            else
            {
#pragma unroll
                for(int k = 0; k < 4; ++k)
                {
                    I8 q = fragment<16>(qs, head, k * 128 + group * 32);
                    a    = mma(fragment<128>(kvs, wave * 32 + head, k * 128 + group * 32), q, a);
                    b = mma(fragment<128>(kvs, wave * 32 + 16 + head, k * 128 + group * 32), q, b);
                }
            }
            float scale = p.q_scale * p.kv_scale * p.attention_scale;
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                if constexpr(IdsMode == 0)
                {
                    a[j] = selected[wave * 32 + group * 4 + j] >= 0 ? a[j] * scale : -INFINITY;
                    b[j] = selected[wave * 32 + 16 + group * 4 + j] >= 0 ? b[j] * scale : -INFINITY;
                }
                else
                {
                    a[j] = ((valid_bits >> (group * 4 + j)) & 1) ? a[j] * scale : -INFINITY;
                    b[j] = ((valid_bits >> (16 + group * 4 + j)) & 1) ? b[j] * scale : -INFINITY;
                }
            }
            float m = -INFINITY;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                m = fmaxf(m, fmaxf(a[j], b[j]));
#if SM_LANE_MODE & 1
            if constexpr(Splits == 8 && Order == 0 && Prefetch && !TailBarrier && Waves == 4)
                m = softmax_four_groups<true>(m);
            else
            {
                m = fmaxf(m, __shfl_xor(m, 16));
                m = fmaxf(m, __shfl_xor(m, 32));
            }
#else
            m = fmaxf(m, __shfl_xor(m, 16));
            m = fmaxf(m, __shfl_xor(m, 32));
#endif
            if(group == 0)
                exchange[wave * 16 + head] = m;
        }
        __syncthreads();
        float m = 0.f, inverse = 0.f, tile_lse = -INFINITY, online_alpha = 0.f;
        if constexpr(LocalReduce)
        {
            if(wave < 4)
                m = fmaxf(fmaxf(exchange[head], exchange[16 + head]),
                          fmaxf(exchange[32 + head], exchange[48 + head]));
        }
        else
        {
            if(wave == (Waves == 8 ? 4 : 0) && lane < 16)
                row_max[lane] = fmaxf(fmaxf(exchange[lane], exchange[16 + lane]),
                                      fmaxf(exchange[32 + lane], exchange[48 + lane]));
            __syncthreads();
        }
        if(wave < 4)
        {
            if constexpr(!LocalReduce)
                m = row_max[head];
            if constexpr(Online)
            {
                const float newmax = fmaxf(merge_max, m);
                online_alpha = merge_max == -INFINITY ? 0.f : exp2f((merge_max - newmax) * LOG2E);
                merge_max    = newmax;
                m            = newmax;
            }
            float z = 0.f;
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                a[j] = m == -INFINITY ? 0.f : exp2f((a[j] - m) * LOG2E);
                b[j] = m == -INFINITY ? 0.f : exp2f((b[j] - m) * LOG2E);
                z += a[j] + b[j];
            }
#if SM_LANE_MODE & 2
            if constexpr(Splits == 8 && Order == 0 && Prefetch && !TailBarrier && Waves == 4)
                z = softmax_four_groups<false>(z);
            else
            {
                z += __shfl_xor(z, 16);
                z += __shfl_xor(z, 32);
            }
#else
            z += __shfl_xor(z, 16);
            z += __shfl_xor(z, 32);
#endif
            if(group == 0)
            {
                if constexpr(LocalReduce)
                    exchange_sum[wave * 16 + head] = z;
                else
                    exchange[wave * 16 + head] = z;
            }
        }
        __syncthreads();
        if constexpr(LocalReduce)
        {
            if(wave < 4)
            {
                float z = (exchange_sum[head] + exchange_sum[16 + head]) +
                          (exchange_sum[32 + head] + exchange_sum[48 + head]);
                if constexpr(Online)
                    merge_sum = merge_sum * online_alpha + z;
                else
                {
                    inverse  = z > 0 ? 1.f / z : 0.f;
                    tile_lse = z > 0 ? m + log2f(z) * LN2 : -INFINITY;
                }
            }
        }
        else
        {
            if(wave == (Waves == 8 ? 4 : 0) && lane < 16)
            {
                float z = (exchange[lane] + exchange[16 + lane]) +
                          (exchange[32 + lane] + exchange[48 + lane]);
                row_inv[lane] = z > 0 ? 1.f / z : 0.f;
                p.partial_lse[(int64_t(query) * 17 + split) * 64 + hbase + lane] =
                    z > 0 ? row_max[lane] + log2f(z) * LN2 : -INFINITY;
            }
            __syncthreads();
        }
        if(wave < 4)
        {
            float inv   = (Online ? 1.f : (LocalReduce ? inverse : row_inv[head])) / p.p_scale;
            uint32_t pa = pack4(a * inv), pb = pack4(b * inv);
            *reinterpret_cast<uint32_t*>(ps + p_address(head, wave * 32 + group * 4))      = pa;
            *reinterpret_cast<uint32_t*>(ps + p_address(head, wave * 32 + 16 + group * 4)) = pb;
        }
        __syncthreads();
        if(wave < 4)
        {
            float alpha = 0.f, beta = 0.f;
            if constexpr(!Online)
            {
                if constexpr(Tiles > 1)
                {
                    if(iteration == 0)
                    {
                        merge_max = tile_lse;
                        merge_sum = tile_lse == -INFINITY ? 0.f : 1.f;
                    }
                    else
                    {
                        float delta = exp2f(-fabsf((merge_max == -INFINITY && tile_lse == -INFINITY)
                                                       ? 0.f
                                                       : merge_max - tile_lse) *
                                            LOG2E);
                        alpha =
                            merge_max == -INFINITY ? 0.f : (merge_max >= tile_lse ? 1.f : delta);
                        beta = tile_lse == -INFINITY ? 0.f : (tile_lse >= merge_max ? 1.f : delta);
                        merge_sum = merge_sum * alpha + beta;
                        merge_max = fmaxf(merge_max, tile_lse);
                    }
                }
                else
                {
                    merge_max = tile_lse;
                    merge_sum = tile_lse == -INFINITY ? 0.f : 1.f;
                }
            }
            I8 prob = p_fragment(ps, head, group * 32);
            if constexpr(Online)
            {
                static_assert(PVGroup == 1 && LocalReduce);
                if constexpr(PadFactored)
                    padded_online_pv_step<0, PVVGPR>(kvs, prob, wave, online_alpha, merged);
                else
                {
#pragma unroll
                    for(int tile = 0; tile < 8; ++tile)
                    {
                        I8 v         = v_fragment_tr(kvs, wave * 128 + tile * 16);
                        merged[tile] = online_pv_mma<PVVGPR>(v, prob, merged[tile] * online_alpha);
                    }
                }
            }
            else if constexpr(PVGroup == 4)
            {
#pragma unroll
                for(int base_tile = 0; base_tile < 8; base_tile += 4)
                {
                    I8 values[4];
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        values[j] = pv_pipe_fragment(kvs, wave * 128 + (base_tile + j) * 16);
                    // Explicit register inputs/outputs prevent wait moving past MFMA.
                    pv_pipe_wait(values);
                    F4 products[4];
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        products[j] = mma(values[j], prob, F4{});
                    asm volatile(""
                                 : "+v"(products[0]),
                                   "+v"(products[1]),
                                   "+v"(products[2]),
                                   "+v"(products[3]));
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                    {
                        const int tile = base_tile + j;
                        F4 value       = products[j] * (p.p_scale * p.kv_scale);
                        if(iteration == 0)
                            merged[tile] = value;
                        else
                        {
#pragma unroll
                            for(int k = 0; k < 4; ++k)
                                merged[tile][k] = fmaf(value[k], beta, merged[tile][k] * alpha);
                        }
                    }
                }
            }
            else
            {
                if constexpr(PadFactored)
                {
                    padded_pv_step<0>(
                        kvs, prob, wave, iteration, alpha, beta, p.p_scale * p.kv_scale, merged);
                }
                else
                {
#pragma unroll
                    for(int tile = 0; tile < 8; ++tile)
                    {
                        I8 v;
                        if constexpr(Transpose)
                            v = v_fragment_tr(kvs, wave * 128 + tile * 16);
                        else
                            v = v_fragment(kvs, wave * 128 + tile * 16 + head, group * 32);
                        F4 out = mma(v, prob, F4{});
                        out *= p.p_scale * p.kv_scale;
                        if(iteration == 0)
                            merged[tile] = out;
                        else
                        {
#pragma unroll
                            for(int j = 0; j < 4; ++j)
                                merged[tile][j] = fmaf(out[j], beta, merged[tile][j] * alpha);
                        }
                    }
                }
            }
        }
        if constexpr(TailBarrier && Tiles > 1)
            __syncthreads();
    }
    if(wave < 4)
    {
        float inv = merge_sum > 0 ? (Online ? p.p_scale * p.kv_scale : 1.f) / merge_sum : 0.f;
#pragma unroll
        for(int tile = 0; tile < 8; ++tile)
        {
            int64_t off =
                partial_address<PartialTile>(((int64_t(query) * Splits + part) * 64 + hbase + head),
                                             wave * 128 + tile * 16 + group * 4);
            partial_store<NT>(p.partial_o + off, merged[tile] * inv);
        }
        if(wave == 0 && group == 0)
            p.partial_lse[(int64_t(query) * Splits + part) * 64 + hbase + head] =
                merge_sum > 0 ? merge_max + log2f(merge_sum) * LN2 : -INFINITY;
    }
}

template <bool Permute, bool Max>
__device__ __forceinline__ float split_reduce(float x)
{
    if constexpr(Permute)
    {
        // Rotate each 16-lane row by eight == lane xor 8. These builtins
        // expose dependencies to LLVM's scheduling/hazard handling.
        float a = __builtin_bit_cast(
            float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0x128, 0xf, 0xf, false));
        x = Max ? fmaxf(x, a) : x + a;
        // reducer. Pair-valued builtins lost a half under this compiler.
        uint32_t seed16, bits16 = __builtin_bit_cast(uint32_t, x);
        asm("v_mov_b32_e32 %0, %1" : "=v"(seed16) : "v"(bits16));
        asm("s_nop 2\n\tv_permlane16_swap_b32 %0, %1" : "+v"(seed16), "+v"(bits16));
        float l16 = __builtin_bit_cast(float, seed16), r16 = __builtin_bit_cast(float, bits16);
        x = Max ? fmaxf(l16, r16) : l16 + r16;
        // Seed both halves: one result is self, the other is xor-32 partner.
        uint32_t seed32, bits32 = __builtin_bit_cast(uint32_t, x);
        asm("v_mov_b32_e32 %0, %1" : "=v"(seed32) : "v"(bits32));
        asm("s_nop 2\n\tv_permlane32_swap_b32 %0, %1" : "+v"(seed32), "+v"(bits32));
        float l = __builtin_bit_cast(float, seed32), r = __builtin_bit_cast(float, bits32);
        return Max ? fmaxf(l, r) : l + r;
    }
    else
    {
#pragma unroll
        for(int d = 8; d < 64; d *= 2)
        {
            float a = __shfl_xor(x, d, 64);
            x       = Max ? fmaxf(x, a) : x + a;
        }
        return x;
    }
}

// One common BF16 merge body; Parallel changes the split reduction tree only.
__device__ __forceinline__ void store_bf16(uint8_t* out, int64_t offset, F4 value)
{

    uint32_t lo, hi;
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(lo) : "v"(value[0]), "v"(value[1]));
    asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(hi) : "v"(value[2]), "v"(value[3]));
    *reinterpret_cast<I2*>(reinterpret_cast<uint16_t*>(out) + offset) = I2{int(lo), int(hi)};
}
template <int Tile, bool NT>
__device__ __forceinline__ F4 load_partial(const Params& p, int64_t row, int col)
{
    return partial_load<NT>(p.partial_o + partial_address<Tile>(row, col));
}
template <int Splits, int PartialTile, bool Remap, bool NT, bool Parallel>
__global__ __launch_bounds__((Parallel ? 4 : 1) * 64) void merge(Params p)
{
    const int lane = threadIdx.x & 63, wave = threadIdx.x / 64;
    if constexpr(Parallel)
    {
        constexpr int R    = (Splits + 7) / 8;
        const CtaTile work = cta_tile<64, 4, Remap>();
        const int split = lane / 8, col = work.c * 128 + wave * 32 + (lane % 8) * 4, head = work.b;
        const int64_t row = int64_t(work.q) * 64 + head;
        float ls[R];
        F4 values[R];
        float m = -INFINITY;
#pragma unroll
        for(int r = 0; r < R; ++r)
        {
            const int s        = split + r * 8;
            const int64_t stat = (int64_t(work.q) * Splits + s) * 64 + head;
            ls[r]              = s < Splits ? p.partial_lse[stat] : -INFINITY;
            values[r]          = s < Splits ? load_partial<PartialTile, NT>(p, stat, col) : F4{};
            m                  = fmaxf(m, ls[r]);
        }
        m = split_reduce<true, true>(m);
        float weights[R];
        float z = 0.f;
#pragma unroll
        for(int r = 0; r < R; ++r)
        {
            weights[r] = ls[r] == -INFINITY ? 0.f : exp2f((ls[r] - m) * LOG2E);
            z += weights[r];
        }
        z = split_reduce<true, false>(z);
        F4 out{};
        const float inv = z > 0 ? 1.f / z : 0.f;
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            float x = 0.f;
#pragma unroll
            for(int r = 0; r < R; ++r)
                x += weights[r] > 0 ? values[r][j] * weights[r] : 0.f;
            out[j] = split_reduce<true, false>(x) * inv;
        }
        if(split == 0)
            store_bf16(p.out, row * 512 + col, out);
        if(p.lse && work.c == 0 && threadIdx.x == 0)
            p.lse[row] = z > 0 ? m + log2f(z) * LN2 : -INFINITY;
    }
    else
    {
        const CtaTile work = cta_tile<4, 32, Remap>();
        const int head = work.b * 16 + lane / 4, col = work.c * 16 + (lane % 4) * 4;
        const int64_t row = int64_t(work.q) * 64 + head;
        float ls[Splits], m = -INFINITY, z = 0.f;
        F4 accum{};
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            ls[s] = p.partial_lse[(int64_t(work.q) * Splits + s) * 64 + head];
            m     = fmaxf(m, ls[s]);
        }
#pragma unroll
        for(int s = 0; s < Splits; ++s)
        {
            float w = m == -INFINITY ? 0.f : exp2f((ls[s] - m) * LOG2E);
            z += w;
            F4 value =
                load_partial<PartialTile, NT>(p, (int64_t(work.q) * Splits + s) * 64 + head, col);
#pragma unroll
            for(int j = 0; j < 4; ++j)
                accum[j] = fmaf(w > 0 ? value[j] : 0.f, w, accum[j]);
        }
        float inv = z > 0 ? 1.f / z : 0.f;
        store_bf16(p.out, row * 512 + col, accum * inv);
        if(p.lse && work.c == 0 && lane % 4 == 0)
            p.lse[row] = z > 0 ? m + log2f(z) * LN2 : -INFINITY;
    }
}

template <bool BF16, bool Unit>
hipError_t launch(Params p, hipStream_t stream)
{
    hipLaunchKernelGGL((attention<17,
                                  0,
                                  true,
                                  false,
                                  4,
                                  false,
                                  BF16,
                                  Unit,
                                  16,
                                  true,
                                  true,
                                  1,
                                  0,
                                  false,
                                  true,
                                  true>),
                       dim3(p.queries, 17, 4),
                       dim3(256),
                       0,
                       stream,
                       p);
    auto status = hipGetLastError();
    if(status != hipSuccess)
        return status;
    hipLaunchKernelGGL(
        (merge<17, 16, true, true, false>), dim3(p.queries, 4, 32), dim3(64), 0, stream, p);
    return hipGetLastError();
}
} // namespace sparse_mla_fp8::small
namespace sparse_mla_fp8 {
hipError_t launch_small(Params p, hipStream_t stream)
{
    if(!p.bf16_q)
        return small::launch<false, false>(p, stream);
    if(p.q_scale == 1.f)
        return small::launch<true, true>(p, stream);
    return small::launch<true, false>(p, stream);
}
} // namespace sparse_mla_fp8
