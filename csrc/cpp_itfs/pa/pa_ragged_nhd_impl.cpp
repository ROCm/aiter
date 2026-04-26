#include <__clang_hip_runtime_wrapper.h>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

// using float16x4 = __fp16 __attribute__ ((ext_vector_type(2)));
// using float16x8 = __fp16 __attribute__ ((ext_vector_type(8)));
// using float16x32 = __fp16 __attribute__ ((ext_vector_type(32)));
using bfloat16x2 = __bf16  __attribute__ ((ext_vector_type(2)));
using bfloat16x4 = __bf16  __attribute__ ((ext_vector_type(4)));
using bfloat16x8 = __bf16  __attribute__ ((ext_vector_type(8)));
using bfloat16x32 = __bf16 __attribute__ ((ext_vector_type(32)));
using float32x16 = float  __attribute__ ((ext_vector_type(16)));
using float32x4 = float  __attribute__ ((ext_vector_type(4)));
using float32x2 = float  __attribute__ ((ext_vector_type(2)));
using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
using int32x16_t = int  __attribute__ ((ext_vector_type(16)));
using uint32x2_t = uint  __attribute__ ((ext_vector_type(2)));

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
union BufferResource {
    __device__ __inline__ constexpr BufferResource()
        : config(0x00020000U) {}

    __device__ __inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
        : address(buffer_address),
          range(buffer_size),
          config(0x00020000U) {}
    
    __device__ __inline__ constexpr void set_base(void* buffer_address) {
        address = buffer_address;
    }

    int32x4_t descriptor;
    struct{
        void* address;      // 8B, out of which first 48b is address, and 16b is stride (unused)
        uint32_t range;     // Byte range for the buffer resource
        uint32_t config;    // Constant, DFMT=32b
    };
};

using DWORDX4 = int32x4_t;

#define NUM_THREADS 256

template <typename Object, std::enable_if_t<std::is_trivially_copyable_v<Object>, int> = 0>
__device__ inline auto amd_wave_read_first_lane(const Object& obj)
{
    constexpr size_t ObjectSize = sizeof(Object);
    constexpr size_t SGPR_size  = 4;
    constexpr size_t NumFull    = ObjectSize / SGPR_size;
    constexpr size_t Tail       = ObjectSize % SGPR_size;

    const unsigned char* src = reinterpret_cast<const unsigned char*>(&obj);
    alignas(Object) unsigned char dst[ObjectSize];

    #pragma unroll
    for (size_t Ic = 0; Ic < NumFull; Ic++) {
        size_t offset = Ic * SGPR_size;
        uint32_t read_src;
        __builtin_memcpy(&read_src, src + offset, SGPR_size);
        read_src = __builtin_amdgcn_readfirstlane(read_src);
        __builtin_memcpy(dst + offset, &read_src, SGPR_size);
    }

    if constexpr(Tail != 0)
    {
        constexpr size_t offset = NumFull * SGPR_size;
        uint32_t tail_loc       = 0;
        __builtin_memcpy(&tail_loc, src + offset, Tail);
        tail_loc = __builtin_amdgcn_readfirstlane(tail_loc);
        __builtin_memcpy(dst + offset, &tail_loc, Tail);
    }
    Object out;
    __builtin_memcpy(&out, dst, ObjectSize);
    return out;
}

__device__ DWORDX4 read_dwordx4(BufferResource& buffer, int soffset, int nstride) {
    DWORDX4 data;
    //volatile int dummy[40] = {0};
    int lane_offset = threadIdx.x * sizeof(DWORDX4);
    int soff = soffset;
    auto r = amd_wave_read_first_lane(buffer.descriptor);
    asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(data)
        :[vaddr]"v"(lane_offset), [srsrc]"s"(r), [soffset]"s"(soff));
    return data;
}

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        auto r = amd_wave_read_first_lane(buffer.descriptor);
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(voffset), [srsrc]"s"(r), [soffset]"s"(soffset));
    } else {
        v = *(T*)((char*)buffer.address + soffset + voffset);
    }
    return v;
}

__device__ float llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int voffset,
                                 int soffset,
                                 int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ __inline__
void llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,    //
                                as3_uint32_ptr lds_ptr, // LDS base offset
                                int32_t size,           // Data byte size: 1/2/4 (/12/16 for gfx950)
                                int32_t voffset,        // voffset(VGPR, included in bounds checking and swizzling)
                                int32_t soffset,        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
                                int32_t offset,         // imm offset(imm, included in bounds checking and swizzling)
                                int32_t aux             // auxiliary/cachepolicy(imm):
                            ) __asm("llvm.amdgcn.raw.buffer.load.lds");

template<typename T>
__device__ T buffer_load_dword(BufferResource& buffer, int soffset, int voffset, int coffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        r.x         = __builtin_amdgcn_readfirstlane(r.x);
        r.y         = __builtin_amdgcn_readfirstlane(r.y);
        r.z         = __builtin_amdgcn_readfirstlane(r.z);
        r.w         = __builtin_amdgcn_readfirstlane(r.w);

        //auto r = amd_wave_read_first_lane(buffer.descriptor);
        asm volatile("buffer_load_dword %[vdst], %[vaddr], %[srsrc], 0 offen offset:%[coffset]\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(voffset), [srsrc]"s"(r), [coffset]"n"(coffset)
            : "memory");
    } else {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        auto d = llvm_amdgcn_raw_buffer_load_fp32(r, voffset + coffset, soffset, 0);
        v = __builtin_bit_cast(T, d);
        //v = *(T*)((char*)buffer.address + soffset + voffset + coffset);
    }
    return v;
}

template<typename T>
__device__ T global_load_dwordx4(void* addr, uint64_t voffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        asm volatile("global_load_dwordx4 %[vdst], %[vaddr], off\n"
                :[vdst]"=v"(v)
                :[vaddr]"v"((char*)addr + voffset));
    } else {
        v = *(T*)((char*)addr + voffset);
    }
    return v;
}

__device__ void amdgcn_mfma_f32_16x16x16bf16(bfloat16x4 a, bfloat16x4 b, float32x4& c) {
    c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    // asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %3\n"
    //             : "+v"(c)
    //             : "v"(a), "v"(b), "v"(c)
    //             :);
}

template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}
template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}

#ifndef HQ
#define HQ 32 
#define HK 4
// #define HQ 12
// #define HK 2
#define S 256
#define BLOCK_SIZE 1
#define SCALE 1.0 
#define KV_PART_SIZE 256
#endif
#define KV_MIN_PART_SIZE 256

// This file is tuned for S=256 K/V tiling and write-back; do not mix S=128 when matching PyTorch reference.
static_assert(S == 256, "pa.cpp: K/V LDS and fused write-back are only implemented for S=256");

template<typename TS, typename TD>
TD& cast(TS& t) {
    return (TD&)t;
}


template<typename F, size_t... Is>
constexpr void compile_time_loop_impl(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
}

template<size_t N, typename F>
constexpr void compile_time_loop(F&& f) {
    compile_time_loop_impl(std::forward<F>(f), 
                          std::make_index_sequence<N>{});
}

#define KV_PART_SIZE_WARP (KV_MIN_PART_SIZE / 4)

struct share_buf_t {
    __bf16* buf;
    __device__ share_buf_t(__bf16* buf) : buf(buf) {}
      // for output reduce: 4 warpx16xS
    __device__ __bf16* get_out_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // for key cache: 4 warpx16xS
    __device__ __bf16* get_key_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // for value cache: 4 warpx16xS
    __device__ __bf16* get_value_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // 4warp * (16m + 1)
    __device__ float* get_max_buf(uint m, uint warp_id) {
        return (float*)buf + m + warp_id * 16;
    }
    __device__ float* get_sum_buf(uint m, uint warp_id) {
        return (float*)buf + m + warp_id * 16 + 64;
    }
    __device__ uint* get_idx_buf(uint warp_id) {
        return (uint*)get_key_buf(warp_id);
    }
};

__global__ void __launch_bounds__(NUM_THREADS, 2) pa(
            __bf16* query,          // [B, HQ, S]
            __bf16* key_cache,      // [BLOCK, BLOCK_SIZE, HK, S]
            __bf16* value_cache,
            uint* kv_indptr,        // [B + 1]
            uint* kv_page_indices,  // [B * KV_LEN + 1]
            __bf16* out_seg,        // [B, HQ, PART, S]
            float* qk_ptr,          // [B, HK, HQ // HK, PART * KV_PART_SIZE]
            float* max_out,         // [B, HQ, PART, 1]
            float* sum_out) {       // [B, HQ, PART, 1]
    // wg: B, HK, kv_part
    uint b = blockIdx.x;
    uint hk = blockIdx.y;
    uint hq = hk * (HQ / HK);
    uint kv_part = blockIdx.z;
    uint lane_id = threadIdx.x % 64;
    uint warp_id = threadIdx.x / 64;
    const uint q_b_stride = HQ * S;
    const uint q_h_stride = S;
    query += b * q_b_stride + hq * q_h_stride;
    key_cache += hk * S;
    value_cache += hk * S;
    uint last_kv_idx = kv_indptr[b + 1];
    uint kv_len = last_kv_idx - kv_indptr[b];
    if (kv_part * KV_PART_SIZE >= kv_len) return;
    uint kv_len_start = std::min(kv_part * KV_PART_SIZE + warp_id * KV_PART_SIZE_WARP, kv_len);
    uint kv_len_end = std::min(kv_part * KV_PART_SIZE + KV_PART_SIZE, kv_len);
    kv_page_indices += kv_indptr[b];
    out_seg += b * HQ * gridDim.z * S + hq * gridDim.z * S + kv_part * S;
    max_out += b * HQ * gridDim.z * 1 + hq * gridDim.z * 1 + kv_part * 1;
    sum_out += b * HQ * gridDim.z * 1 + hq * gridDim.z * 1 + kv_part * 1;

    // stage1(q*k): [16, 32]x[64, 32]'
    BufferResource q_buf(query, (HQ / HK) * S * sizeof(__bf16));
    static_assert(HQ / HK <= 16, "use mfma16 requires M <= 16");
    bfloat16x8 q_cur[S / 32];
    // key load layout: 4rows x 16 cols
    uint key_load_col_id = lane_id % 16; // 0 ~ 15
    uint key_load_row_id = lane_id / 16; // 0 ~ 3
    // mfma16x16 layout: 16rows x 4cols
    uint fma_col_id = lane_id / 16;      // 0 ~ 3
    uint fma_row_id = lane_id % 16;      // 0 ~ 15
    float prev_max = - FLT_MAX;
    float prev_sum = 0;

    // query -> reg
    if (fma_row_id < HQ / HK) {
        #pragma unroll
        for (uint k = 0; k < S; k += 32) {
#if FAKE_Q
            q_cur[k / 32] = __bf16(1.0f);
#else
            q_cur[k / 32] = buffer_load_dwordx4<bfloat16x8>(q_buf, k * sizeof(__bf16), (fma_row_id * q_h_stride + fma_col_id * 8) * sizeof(__bf16), false);
#endif
        }
    }
    //s_waitcnt_vmcnt<0>();
    __shared__ __bf16 buff_lds[16 * S * 4];
    share_buf_t share_buf(buff_lds);
    float32x4 vout[S / 64 * 4] = {0};
    auto cur_max = prev_max;
    float cur_sum;

    BufferResource idx_buf(kv_page_indices, kv_len * sizeof(uint));
    uint k_idxs[KV_PART_SIZE_WARP / 4];
    auto cur_kv_len_start = kv_len_start + 0 * KV_MIN_PART_SIZE;

    uint* idx_lds = share_buf.get_idx_buf(warp_id);
    uint idx_global = lane_id + cur_kv_len_start;
    idx_lds[lane_id] = (idx_global < kv_len_end) ? kv_page_indices[idx_global] : 0;
    // Keep commented: original raw buffer load to LDS path.
    // On ROCm 7.2 this path appears ineffective (e.g. hipcc -O2 -c -std=c++20 --offload-arch=gfx942 -S -emit-llvm pa.cpp -o k72.ll):
    // LLVM IR does not emit @llvm.amdgcn.raw.buffer.load.lds.
    //llvm_amdgcn_raw_buffer_load_lds(idx_buf.descriptor, (as3_uint32_ptr)idx_lds, 4, (lane_id + cur_kv_len_start) * sizeof(uint), 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (uint part_idx = 0; part_idx < KV_PART_SIZE / KV_MIN_PART_SIZE; part_idx++) {
        if (part_idx == 0)
        if (BLOCK_SIZE == 1) {
            for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
    // #if FAKE_K_IDX // unused debug path
    //             uint global_row_id = n * 4 + key_load_row_id + cur_kv_len_start;
    //             global_row_id = global_row_id < kv_len_end ? global_row_id : 0;
    //             k_idxs[n] = global_row_id + 1 + kv_indptr[b];
    // #else
                //k_idxs[n] = buffer_load_dword<uint>(idx_buf, 0, (key_load_row_id + cur_kv_len_start) * sizeof(uint), n * 4 * sizeof(uint));
            k_idxs[n] = idx_lds[n * 4 + key_load_row_id];
    // #endif
            }
        }
        // key -> reg
        float32x4 acc[KV_PART_SIZE_WARP / 16] = {0};

        bfloat16x8 k_reg_caches[KV_PART_SIZE_WARP / 4], k_reg_hi[KV_PART_SIZE_WARP / 4];
        bfloat16x8 v_reg_caches[KV_PART_SIZE_WARP / 4], v_reg_hi[KV_PART_SIZE_WARP / 4];

        if (part_idx == 0) {
            for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
                auto offset = k_idxs[n] * (HK * S * sizeof(__bf16)) + 8 * sizeof(__bf16) * key_load_col_id;
                k_reg_caches[n] = global_load_dwordx4<bfloat16x8>(key_cache, offset);
                k_reg_hi[n] = global_load_dwordx4<bfloat16x8>(key_cache, offset + 128 * sizeof(__bf16));
            }
        }
        for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
            auto offset = k_idxs[n] * (HK * S * sizeof(__bf16)) + 8 * sizeof(__bf16) * key_load_col_id;
            v_reg_caches[n] = global_load_dwordx4<bfloat16x8>(value_cache, offset, false);
            v_reg_hi[n] = global_load_dwordx4<bfloat16x8>(value_cache, offset + 128 * sizeof(__bf16), false);
        }
        __bf16* cur_k_buff_lds = share_buf.get_key_buf(warp_id);
        //4 times
        for (uint n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
            // swizzle: row ^ col
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 0 * 4) * S + (key_load_col_id ^ (key_load_row_id + 0)) * 8]) = k_reg_caches[4 * n + 0];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 1 * 4) * S + (key_load_col_id ^ (key_load_row_id + 4)) * 8]) = k_reg_caches[4 * n + 1];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 2 * 4) * S + (key_load_col_id ^ (key_load_row_id + 8)) * 8]) = k_reg_caches[4 * n + 2];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 3 * 4) * S + (key_load_col_id ^ (key_load_row_id + 12))* 8]) = k_reg_caches[4 * n + 3];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 0 * 4) * S + 128 + (key_load_col_id ^ (key_load_row_id + 0)) * 8]) = k_reg_hi[4 * n + 0];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 1 * 4) * S + 128 + (key_load_col_id ^ (key_load_row_id + 4)) * 8]) = k_reg_hi[4 * n + 1];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 2 * 4) * S + 128 + (key_load_col_id ^ (key_load_row_id + 8)) * 8]) = k_reg_hi[4 * n + 2];
            *(bfloat16x8*)(&cur_k_buff_lds[(key_load_row_id + 3 * 4) * S + 128 + (key_load_col_id ^ (key_load_row_id + 12))* 8]) = k_reg_hi[4 * n + 3];

            for (int k = 0; k < S / 32; k++) {
                const uint half = (k < S / 64) ? 0u : 128u;
                const uint kk = k & (S / 64 - 1);
                auto k_cur = *(bfloat16x8*)(&cur_k_buff_lds[fma_row_id * S + half + (fma_row_id ^ (fma_col_id + 4 * kk)) * 8]);
                amdgcn_mfma_f32_16x16x16bf16(k_cur.lo, q_cur[k].lo, acc[n]);
                amdgcn_mfma_f32_16x16x16bf16(k_cur.hi, q_cur[k].hi, acc[n]);
            }
            acc[n] *= SCALE;
        }
    #if OUTPUT_QK
        // for debug
        if (qk_ptr) {
            // [B, HK, HQ / HK, stride]
            auto stride = (kv_len + KV_PART_SIZE - 1) / KV_PART_SIZE * KV_PART_SIZE;
            if (fma_row_id < HQ / HK) {
                for (int n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
                    auto cur_tmp_ptr = qk_ptr + b * HK * HQ / HK * stride + (hk * (HQ / HK) + fma_row_id) * stride + cur_kv_len_start + n * 16 + fma_col_id * 4;
                    for (int i = 0; i < 4; i++)
                        cur_tmp_ptr[i] = acc[n][i];
                }
            }
        }
    #endif

        auto cur_kv_len_start_copy = cur_kv_len_start;
        // --------------------------
        cur_kv_len_start += KV_MIN_PART_SIZE;
        if (part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1)
        if (BLOCK_SIZE == 1) {
            // llvm_amdgcn_raw_buffer_load_lds(idx_buf.descriptor, (as3_uint32_ptr)idx_lds, 4, (lane_id + kv_len_start) * sizeof(uint), (part_idx + 1) * KV_MIN_PART_SIZE * sizeof(uint), 0, 0);
            for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
    #if FAKE_K_IDX
                uint global_row_id = n * 4 + key_load_row_id + cur_kv_len_start;
                global_row_id = global_row_id < kv_len_end ? global_row_id : 0;
                k_idxs[n] = global_row_id + 1;
    #else
                k_idxs[n] = buffer_load_dword<uint>(idx_buf, 0, (key_load_row_id + cur_kv_len_start) * sizeof(uint), n * 4 * sizeof(uint));
                // k_idxs[n] = idx_lds[n * 4 + key_load_row_id];
    #endif
            }
        }
        // -----------------------------
        // sum(exp(acc-max))
        for (uint n = 0; n < 4; n++) {
            for (uint i = 0; i < 4; i++) {
                auto tmp = n * 16 + fma_col_id * 4 + i + cur_kv_len_start_copy < kv_len_end ? acc[n][i] : -FLT_MAX;
                cur_max = fmaxf(cur_max, tmp);
            }
        }
        for(int mask = 64 / 2; mask >= 16; mask /= 2) {
            cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
        }
        cur_sum = 0;
        for (uint n = 0; n < 4; n++) {
            for (uint i = 0; i < 4; i++) {
                acc[n][i] = n * 16 + fma_col_id * 4 + i + cur_kv_len_start_copy < kv_len_end ? __expf(acc[n][i] - cur_max) : 0;
                cur_sum += acc[n][i];
            }
        }
        for(int mask = 64 / 2; mask >= 16; mask /= 2) {
            cur_sum += __shfl_xor(cur_sum, mask);
        }
        // ----------------------------------
        if (part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1) {
            for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
                auto offset = k_idxs[n] * (HK * S * sizeof(__bf16)) + 8 * sizeof(__bf16) * key_load_col_id;
                k_reg_caches[n] = global_load_dwordx4<bfloat16x8>(key_cache, offset);
                k_reg_hi[n] = global_load_dwordx4<bfloat16x8>(key_cache, offset + 128 * sizeof(__bf16));
            }
        }
        // ----------------------------------
        auto fixed_sum = prev_sum * __expf(prev_max - cur_max);
        cur_sum += fixed_sum;
        const float inv_sum_scale = 1.f / (cur_sum + 1e-6f);
        bfloat16x4 acc_low[KV_PART_SIZE_WARP / 16];
        for (uint n = 0; n < 4; n++) {
            for (uint i = 0; i < 4; i++) {
                auto tmp = acc[n][i] * inv_sum_scale;
                acc_low[n][i] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(tmp) >> 16));
            }
        }
        // compensation prev vout
        if (part_idx != 0) {
            for (int k = 0; k < S / 64 * 4; k++) {
                vout[k] = fixed_sum * inv_sum_scale * vout[k];
            }
        }
        //s_waitcnt_vmcnt<0>();
        __bf16* cur_v_buff_lds = share_buf.get_value_buf(warp_id);
        //__bf16* cur_v_buff_lds = kv_buff_lds + warp_id * 16 * S;
        for (uint n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 0 * 4) * S + key_load_col_id * 8]) = v_reg_caches[4 * n + 0];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 1 * 4) * S + key_load_col_id * 8]) = v_reg_caches[4 * n + 1];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 2 * 4) * S + key_load_col_id * 8]) = v_reg_caches[4 * n + 2];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 3 * 4) * S + key_load_col_id * 8]) = v_reg_caches[4 * n + 3];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 0 * 4) * S + 128 + key_load_col_id * 8]) = v_reg_hi[4 * n + 0];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 1 * 4) * S + 128 + key_load_col_id * 8]) = v_reg_hi[4 * n + 1];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 2 * 4) * S + 128 + key_load_col_id * 8]) = v_reg_hi[4 * n + 2];
            *(bfloat16x8*)(&cur_v_buff_lds[(key_load_row_id + 3 * 4) * S + 128 + key_load_col_id * 8]) = v_reg_hi[4 * n + 3];

            // layout: [4tx4, 16tx4k]-> 16 tokens, 64 of head size used for one iter
            // K/N 0 1 2 3   4 5 6 7   8 9 10 11  12 13 14 15 ...  56 57 58 59  60 61 62 63 
            //   0 thread0   thread1   thread2    thread3          thread14     thread15
            //   1
            //   2
            //   3
            //   4 thread16  thread17  thread18   thread19         thread30     thread31
            //   5
            //   6
            //   7
            //   ...
            //  12 thread48  thread49  thread50   thread51         thread62     thread63
            //  13
            //  14
            //  15
            bfloat16x4 v_curs[4], v_curs_tr[4];
            auto transpose4x4 = [] (bfloat16x4 in[4], bfloat16x4 out[4]) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        out[i][j] = in[j][i];
                }
            };
            for (int k = 0; k < S / 64; k++) {
                v_curs[0] = *(bfloat16x4*)(&cur_v_buff_lds[(key_load_row_id * 4 + 0) * S + k * 64 + key_load_col_id * 4]);
                v_curs[1] = *(bfloat16x4*)(&cur_v_buff_lds[(key_load_row_id * 4 + 1) * S + k * 64 + key_load_col_id * 4]);
                v_curs[2] = *(bfloat16x4*)(&cur_v_buff_lds[(key_load_row_id * 4 + 2) * S + k * 64 + key_load_col_id * 4]);
                v_curs[3] = *(bfloat16x4*)(&cur_v_buff_lds[(key_load_row_id * 4 + 3) * S + k * 64 + key_load_col_id * 4]);
                transpose4x4(v_curs, v_curs_tr);
                amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[0], acc_low[n], vout[k * 4 + 0]);
                amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[1], acc_low[n], vout[k * 4 + 1]);
                amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[2], acc_low[n], vout[k * 4 + 2]);
                amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[3], acc_low[n], vout[k * 4 + 3]);
            }
        }
        prev_max = cur_max;
        prev_sum = cur_sum;
    }
    __syncthreads();
    // cross max/sum
    if (fma_col_id == 0) {
        *share_buf.get_max_buf(fma_row_id, warp_id) = cur_max;
        *share_buf.get_sum_buf(fma_row_id, warp_id) = cur_sum;
    }
    __syncthreads();
    // Begin write-back
    // each wave processes up to ceil(HQ/HK, 4) query tokens (TOKENS_PER_WARP); must match the write-back loop below
    constexpr uint M = HQ / HK; // gqa ratio
    constexpr uint TOKENS_PER_WARP = (M + 3) / 4; // ceil_div(M, 4): query tokens per warp

    float maxs[4 * TOKENS_PER_WARP], sums[4 * TOKENS_PER_WARP];
    float real_max[TOKENS_PER_WARP], real_sum[TOKENS_PER_WARP];

    /*
    // old fixed split (requires HQ/HK divisible by 4):
    float maxs[HQ / HK], sums[HQ / HK];
    float real_max[HQ / HK / 4], real_sum[HQ / HK / 4];
    for (uint i = 0; i < HQ / HK / 4; i++) {
        uint m_token_id = HQ / HK / 4 * warp_id + i;
        maxs[4 * i + 0] = *share_buf.get_max_buf(m_token_id, 0);
        maxs[4 * i + 1] = *share_buf.get_max_buf(m_token_id, 1);
        maxs[4 * i + 2] = *share_buf.get_max_buf(m_token_id, 2);
        maxs[4 * i + 3] = *share_buf.get_max_buf(m_token_id, 3);
        real_max[i] = fmaxf(
            fmaxf(maxs[4 * i + 0], maxs[4 * i + 1]),
            fmaxf(maxs[4 * i + 2], maxs[4 * i + 3]));
        sums[4 * i + 0] = *share_buf.get_sum_buf(m_token_id, 0);
        sums[4 * i + 1] = *share_buf.get_sum_buf(m_token_id, 1);
        sums[4 * i + 2] = *share_buf.get_sum_buf(m_token_id, 2);
        sums[4 * i + 3] = *share_buf.get_sum_buf(m_token_id, 3);
        real_sum[i] = sums[4 * i + 0] * __expf(maxs[4 * i + 0] - real_max[i]) +
                      sums[4 * i + 1] * __expf(maxs[4 * i + 1] - real_max[i]) +
                      sums[4 * i + 2] * __expf(maxs[4 * i + 2] - real_max[i]) +
                      sums[4 * i + 3] * __expf(maxs[4 * i + 3] - real_max[i]);
    }
    */

    for (uint i = 0; i < TOKENS_PER_WARP; i++) {
        uint m_token_id = TOKENS_PER_WARP * warp_id + i;
        if (m_token_id < M) {
            maxs[4 * i + 0] = *share_buf.get_max_buf(m_token_id, 0);
            maxs[4 * i + 1] = *share_buf.get_max_buf(m_token_id, 1);
            maxs[4 * i + 2] = *share_buf.get_max_buf(m_token_id, 2);
            maxs[4 * i + 3] = *share_buf.get_max_buf(m_token_id, 3);
            real_max[i] = fmaxf(
                fmaxf(maxs[4 * i + 0], maxs[4 * i + 1]),
                fmaxf(maxs[4 * i + 2], maxs[4 * i + 3]));

            sums[4 * i + 0] = *share_buf.get_sum_buf(m_token_id, 0);
            sums[4 * i + 1] = *share_buf.get_sum_buf(m_token_id, 1);
            sums[4 * i + 2] = *share_buf.get_sum_buf(m_token_id, 2);
            sums[4 * i + 3] = *share_buf.get_sum_buf(m_token_id, 3);
            real_sum[i] = sums[4 * i + 0] * __expf(maxs[4 * i + 0] - real_max[i]) + 
                          sums[4 * i + 1] * __expf(maxs[4 * i + 1] - real_max[i]) + 
                          sums[4 * i + 2] * __expf(maxs[4 * i + 2] - real_max[i]) + 
                          sums[4 * i + 3] * __expf(maxs[4 * i + 3] - real_max[i]);
        }
    }

    bfloat16x4 vout_low[S / 64 * 4];
    for (uint k = 0; k < S / 64 ; k++) {
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                vout_low[k * 4 + j][i] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(vout[k * 4 + i][j]) >> 16));
    }
    __syncthreads();

    auto shared_out = share_buf.get_out_buf(warp_id) + fma_row_id * S;
    for (int k = 0; k < S / 64; k++) {
        // swizzle: row ^ col
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 0) ^ fma_row_id] = vout_low[k * 4 + 0];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 1) ^ fma_row_id] = vout_low[k * 4 + 1];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 2) ^ fma_row_id] = vout_low[k * 4 + 2];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 3) ^ fma_row_id] = vout_low[k * 4 + 3];
    }
    __syncthreads();
    for (uint i = 0; i < TOKENS_PER_WARP; i++) {
        uint m_token_id = TOKENS_PER_WARP * warp_id + i;
        if (m_token_id < M) {
            const uint idx2 = (m_token_id ^ (lane_id / 2)) * 2 + (lane_id & 1);
            float32x2 out_lo = {};
            float32x2 out_hi = {};
            for (int w = 0; w < 4; w++) {
                bfloat16x2 t_lo = ((bfloat16x2*)(share_buf.get_out_buf(w) + m_token_id * S))[idx2];
                bfloat16x2 t_hi = ((bfloat16x2*)(share_buf.get_out_buf(w) + m_token_id * S + 128))[idx2];
                const float sw = sums[4 * i + w] * __expf(maxs[4 * i + w] - real_max[i]);
                out_lo[0] += (float)t_lo[0] * sw;
                out_lo[1] += (float)t_lo[1] * sw;
                out_hi[0] += (float)t_hi[0] * sw;
                out_hi[1] += (float)t_hi[1] * sw;
            }
            const float inv_sum_scale = 1.f / (real_sum[i] + 1e-6f);
            out_lo[0] *= inv_sum_scale;
            out_lo[1] *= inv_sum_scale;
            out_hi[0] *= inv_sum_scale;
            out_hi[1] *= inv_sum_scale;
            bfloat16x2 tmp_lo;
            tmp_lo[0] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(out_lo[0]) >> 16));
            tmp_lo[1] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(out_lo[1]) >> 16));
            bfloat16x2 tmp_hi;
            tmp_hi[0] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(out_hi[0]) >> 16));
            tmp_hi[1] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(out_hi[1]) >> 16));
            __bf16* row_base = out_seg + m_token_id * gridDim.z * S;
            ((bfloat16x2*)row_base)[lane_id] = tmp_lo;
            ((bfloat16x2*)(row_base + 128))[lane_id] = tmp_hi;
            max_out[m_token_id * gridDim.z] = real_max[i];
            sum_out[m_token_id * gridDim.z] = real_sum[i];
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS, 2) pa_reduce(
            uint* kv_indptr,        // [B + 1]
            __bf16* out_seg,        // [B, HQ, PART, S]
            float* max_out,         // [B, HQ, PART, 1]
            float* sum_out,         // [B, HQ, PART, 1]
            __bf16* out,            // [B, HQ, S]
            uint max_part
            ) {
    // wg: B, HQ
    uint b = blockIdx.x;
    uint hq = blockIdx.y;
    uint lane_id = threadIdx.x % 64;
    uint warp_id = threadIdx.x / 64;
    out += b * HQ * S + hq * S;
    out_seg += b * HQ * max_part * S + hq * max_part * S;
    max_out += b * HQ * max_part * 1 + hq * max_part * 1;
    sum_out += b * HQ * max_part * 1 + hq * max_part * 1;
    uint kv_len = kv_indptr[b + 1] - kv_indptr[b];
    uint part_num = (kv_len + KV_PART_SIZE - 1) / KV_PART_SIZE;
    float real_max = -FLT_MAX;
    for (uint i = lane_id; i < part_num; i += 64) {
        real_max = fmaxf(real_max, max_out[i]);
    }
    for (uint mask = 64 / 2; mask >= 1; mask /= 2) {
        real_max = fmaxf(real_max, __shfl_xor(real_max, mask));
    }

    float32x4 cur = {};
    float warp_sum = 0;
    for (uint i = warp_id; i < part_num; i += 4) {
        auto cur_max = max_out[i];
        auto cur_out_seg = out_seg + i * S;
        auto v = ((bfloat16x4*)cur_out_seg)[lane_id];
        float32x4 tmp;
        tmp[0] = (float)v[0];
        tmp[1] = (float)v[1];
        tmp[2] = (float)v[2];
        tmp[3] = (float)v[3];
        const float w = sum_out[i] * __expf(cur_max - real_max);
        cur[0] += tmp[0] * w;
        cur[1] += tmp[1] * w;
        cur[2] += tmp[2] * w;
        cur[3] += tmp[3] * w;
        warp_sum += sum_out[i] * __expf(cur_max - real_max);
    }

    __shared__ float out_lds[4 * S];
    __shared__ float sum_lds[4];
    if (lane_id == 0) {
        sum_lds[warp_id] = warp_sum;
    }
    if (warp_id != 0) {
        *(float32x4*)(&out_lds[warp_id * S + lane_id * 4 + 0]) = cur;
    }
    __syncthreads();
    if (warp_id == 0) {
        warp_sum += sum_lds[1] + sum_lds[2] + sum_lds[3];
        for (int wi = 1; wi < 4; wi++) {
            cur += *(float32x4*)(&out_lds[wi * S + lane_id * 4 + 0]);
        }
        const float inv_sum_scale = 1.f / (warp_sum + 1e-6f);
        bfloat16x4 o;
        o[0] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(cur[0] * inv_sum_scale) >> 16));
        o[1] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(cur[1] * inv_sum_scale) >> 16));
        o[2] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(cur[2] * inv_sum_scale) >> 16));
        o[3] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(cur[3] * inv_sum_scale) >> 16));
        ((bfloat16x4*)out)[lane_id] = o;
    }
}
