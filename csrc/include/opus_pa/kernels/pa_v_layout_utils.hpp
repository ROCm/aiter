// V_mem_load layout: decode loaded v_regs bytes -> (kv, d) and compare vs HBM shuffled pages.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/kernels/pa_decode_device_utils.hpp"
#include "opus_pa/kernels/pa_mfma_layout_utils.hpp"
#include "opus_pa/kernels/pa_q_gemm_utils.hpp"

namespace pa_decode {

// sp3 prologue: V_buf += tg_idx * stride_kvhead + wave_id * 256 (K only gets tg offset).
__device__ __forceinline__ uint32_t v_pool_lane_base_offset(int lane, int wave, int tg_idx,
                                                            uint32_t stride_kvhead) {
    return v_lane_byte_offset(lane) + static_cast<uint32_t>(tg_idx) * stride_kvhead +
           static_cast<uint32_t>(wave * 256);
}

__device__ __forceinline__ uint32_t k_pool_lane_base_offset(int lane, int tg_idx,
                                                            uint32_t stride_kvhead) {
    return k_lane_byte_offset(lane) + static_cast<uint32_t>(tg_idx) * stride_kvhead;
}

// page_ids slice for pi half-tile (fch 0 -> blocks [0,8), fch 1 -> [8,16)).
template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ const uint32_t* v_mem_load_page_ids_slice(const uint32_t* page_ids,
                                                                     int fch_idx) {
    constexpr int kBlksPerPi = SUB_KV / BLOCK_SIZE / 2;
    return page_ids + fch_idx * kBlksPerPi;
}

template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ int v_mem_load_valid_blks_slice(int valid_blks, int fch_idx) {
    constexpr int kBlksPerPi = SUB_KV / BLOCK_SIZE / 2;
    const int start = fch_idx * kBlksPerPi;
    return (valid_blks > start) ? (valid_blks - start) : 0;
}

// Compute 64-bit load address for one V_mem_load byte (same as v_mem_load_expected_byte).
template<int SUB_KV, int BLOCK_SIZE, int NUM_PAIRS, int IMM_STRIDE>
__device__ __forceinline__ U64 v_mem_load_byte_address(const uint8_t* v_pool,
                                                       const uint32_t* page_ids,
                                                       int valid_blks,
                                                       uint32_t stride_blk,
                                                       uint32_t stride_kvhead,
                                                       int lane,
                                                       int wave,
                                                       int tg_idx,
                                                       int bt_slots,
                                                       int fch_idx,
                                                       int i_idx,
                                                       int byte_b) {
    const int pair = i_idx & 3;
    const uint32_t imm = static_cast<uint32_t>((i_idx >> 2) * IMM_STRIDE);
    const uint32_t* pi_pages = v_mem_load_page_ids_slice<SUB_KV, BLOCK_SIZE>(page_ids, fch_idx);
    const int pi_valid = v_mem_load_valid_blks_slice<SUB_KV, BLOCK_SIZE>(valid_blks, fch_idx);
    U64 v_pool_u = u64_from_ptr(v_pool);
    U64 combined =
        u64_add_imm(v_pool_u, v_pool_lane_base_offset(lane, wave, tg_idx, stride_kvhead));
    U64 v_addrs[NUM_PAIRS];
    v_mem_va_upd<NUM_PAIRS>(v_addrs, combined, pi_pages, lane, wave, bt_slots, pi_valid,
                            stride_blk);
    return u64_add_imm(v_addrs[pair], imm + static_cast<uint32_t>(byte_b));
}

// Expected byte at the same global address V_mem_load would read (mirrors load path).
template<int SUB_KV, int BLOCK_SIZE, int NUM_PAIRS, int IMM_STRIDE>
__device__ __forceinline__ uint8_t v_mem_load_expected_byte(
    const uint8_t* v_pool,
    const uint32_t* page_ids,
    int valid_blks,
    uint32_t stride_blk,
    uint32_t stride_kvhead,
    int block_size,
    int lane,
    int wave,
    int tg_idx,
    int bt_slots,
    int fch_idx,
    int i_idx,
    int byte_b) {
    return *u64_to_ptr(v_mem_load_byte_address<SUB_KV, BLOCK_SIZE, NUM_PAIRS, IMM_STRIDE>(
        v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, lane, wave, tg_idx, bt_slots,
        fch_idx, i_idx, byte_b));
}

// Decode (kv,d) by locating load_ptr within shuffled V pages (exact inverse of gather).
template<int BLOCK_SIZE>
__device__ __forceinline__ bool v_mem_load_decode_from_addr(const uint8_t* v_pool,
                                                            const uint32_t* page_ids,
                                                            int valid_blks,
                                                            uint32_t stride_blk,
                                                            U64 load_addr,
                                                            int& kv_out,
                                                            int& d_out) {
    const uint8_t* load_ptr = u64_to_ptr(load_addr);
    for (int blk = 0; blk < valid_blks; ++blk) {
        const uint8_t* page_base = v_pool + static_cast<size_t>(page_ids[blk]) * stride_blk;
        const uint8_t* page_end = page_base + stride_blk;
        if (load_ptr >= page_base && load_ptr < page_end) {
            const int off = static_cast<int>(load_ptr - page_base);
            const int in_blk = off % BLOCK_SIZE;
            d_out = off / BLOCK_SIZE;
            kv_out = blk * BLOCK_SIZE + in_blk;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ int v_mem_load_bt_slot(int lane, int wave, int pair) {
    const int row_base = lane & ~0xf;
    const int ref_lane = row_base + pair * 4 + 1;
    return static_cast<int>(block_table_lane_index(ref_lane, wave));
}

__device__ __forceinline__ int v_mem_load_page_offset(int lane, int wave, int i_idx, int byte_b,
                                                      int imm_stride) {
    const uint32_t imm = static_cast<uint32_t>((i_idx >> 2) * imm_stride);
    return static_cast<int>(((lane & 0xf) << 4) + imm + static_cast<uint32_t>(byte_b) +
                            static_cast<uint32_t>(wave << 8));
}

// Legacy formula decode (kept for bisect comparison vs addr-inverse).
template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ void v_mem_load_decode_coord_formula(int lane, int wave, int fch_idx,
                                                                 int i_idx, int byte_b,
                                                                 int imm_stride, int& kv_out,
                                                                 int& d_out) {
    constexpr int kBlksPerPi = SUB_KV / BLOCK_SIZE / 2;
    const int pair = i_idx & 3;
    const int page_off = v_mem_load_page_offset(lane, wave, i_idx, byte_b, imm_stride);
    const int in_blk = page_off & (BLOCK_SIZE - 1);
    const int d = page_off / BLOCK_SIZE;
    const int bt_slot = fch_idx * kBlksPerPi + v_mem_load_bt_slot(lane, wave, pair);
    kv_out = bt_slot * BLOCK_SIZE + in_blk;
    d_out = d;
}

__device__ __forceinline__ uint8_t fp8_from_v_reg_dword(uint32_t w, int byte_idx) {
    return static_cast<uint8_t>((w >> (8 * byte_idx)) & 0xffu);
}

// Expected HBM byte for V[kv,d] in tile (shuffled page layout).
__device__ __forceinline__ uint8_t v_hbm_tile_byte(const uint8_t* v_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int kv_head_idx,
                                                   int block_size,
                                                   int kv,
                                                   int d) {
    const int blk = kv / block_size;
    const int in_blk = kv % block_size;
    if (blk >= valid_blks) {
        return 0;
    }
    const uint32_t page = page_ids[blk];
    const size_t kv_head_off =
        static_cast<size_t>(kv_head_idx) * static_cast<size_t>(stride_kvhead);
    const uint8_t* page_base =
        v_pool + static_cast<size_t>(page) * stride_blk + kv_head_off;
    return page_base[v_shuffled_page_offset(in_blk, d, block_size)];
}

// Brute-force (kv,d) by scanning tile HBM — bisect only when addr-inverse fails.
template<int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ bool v_brute_decode_from_hbm(const uint8_t* v_pool,
                                                        const uint32_t* page_ids,
                                                        int valid_blks,
                                                        uint32_t stride_blk,
                                                        int block_size,
                                                        int tile_kv,
                                                        uint8_t got,
                                                        int& kv_out,
                                                        int& d_out,
                                                        int& match_count_out) {
    int match_count = 0;
    int kv_match = -1;
    int d_match = -1;
    for (int kv = 0; kv < tile_kv; ++kv) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            if (v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk, 0, 0, block_size, kv, d) ==
                got) {
                ++match_count;
                kv_match = kv;
                d_match = d;
            }
        }
    }
    match_count_out = match_count;
    if (match_count >= 1) {
        kv_out = kv_match;
        d_out = d_match;
        return true;
    }
    return false;
}

// Gather one MFMA B-operand pair (V[kv,d]) for P@V GEMM1.
// fp8 16x16x32 B[k][n]: n = lane%16, k = 8*(lane/16) + byte_i. Here n = head column
// (head = head_base + wave*16 + lane%16), k = kv contraction (kv = kv_base + 8*(lane/16) + i).
template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE>
__device__ __forceinline__ void v_mfma_gather_b_pair(const uint8_t* v_pool,
                                                     const uint32_t* page_ids,
                                                     int valid_blks,
                                                     uint32_t stride_blk,
                                                     uint32_t stride_kvhead,
                                                     int kv_head_idx,
                                                     int block_size,
                                                     int tile_kv,
                                                     int kv_base,
                                                     int head_base,
                                                     int lane,
                                                     int wave,
                                                     uint32_t& lo,
                                                     uint32_t& hi) {
    const int col = lane & 15;
    const int g = lane >> 4;
    const int d = head_base + (wave << 4) + col;

    uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int kv = kv_base + g * 8 + i;
        bytes[i] = (kv < tile_kv && d < HEAD_DIM)
                       ? v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk, stride_kvhead,
                                         kv_head_idx, block_size, kv, d)
                       : static_cast<uint8_t>(0);
    }
    lo = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    hi = static_cast<uint32_t>(bytes[4]) | (static_cast<uint32_t>(bytes[5]) << 8) |
         (static_cast<uint32_t>(bytes[6]) << 16) | (static_cast<uint32_t>(bytes[7]) << 24);
}

// Extract loaded byte from v_regs for (fch_idx, i_idx, byte_b).
__device__ __forceinline__ uint8_t v_reg_loaded_byte(const uint32_t* v_regs,
                                                     int kv_reg_dwords,
                                                     int fch_idx,
                                                     int i_idx,
                                                     int byte_b) {
    const int dword_idx = byte_b / 4;
    const int byte_in_dw = byte_b % 4;
    const int reg_idx = fch_idx * kv_reg_dwords + i_idx * 4 + dword_idx;
    return fp8_from_v_reg_dword(v_regs[reg_idx], byte_in_dw);
}

// Pack reg location + (kv,d) into one uint32 for bisect dumps.
__device__ __forceinline__ uint32_t v_reg_loc_pack(int fch, int i_idx, int byte_b) {
    return (static_cast<uint32_t>(fch) << 12) | (static_cast<uint32_t>(i_idx) << 4) |
           static_cast<uint32_t>(byte_b);
}

__device__ __forceinline__ uint32_t v_kv_d_pack(int kv, int d) {
    return (static_cast<uint32_t>(kv) & 0xffffu) | (static_cast<uint32_t>(d) << 16);
}

// Compare all bytes this lane loaded vs HBM gather; accumulate into smem[tid*8 + {0..7}].
template<int SUB_KV, int HEAD_DIM, int BLOCK_SIZE, int KV_REG_DWORDS, int LOAD_INSTS, int NUM_PAIRS,
         int IMM_STRIDE>
__device__ __forceinline__ void v_regs_lane_bisect(const uint32_t* v_regs,
                                                   const uint8_t* v_pool,
                                                   const uint32_t* page_ids,
                                                   int valid_blks,
                                                   uint32_t stride_blk,
                                                   uint32_t stride_kvhead,
                                                   int block_size,
                                                   int tile_kv,
                                                   int lane,
                                                   int wave,
                                                   int tg_idx,
                                                   int bt_slots,
                                                   float* smem_stats) {
    int local_addr_mm = 0;
    int local_formula_mm = 0;
    int local_addr_decode_mm = 0;
    int local_brute_mm = 0;
    int local_brute_ambig = 0;
    int local_compared = 0;
    uint32_t first_mismatch_pack = 0xffffffffu;

#pragma unroll
    for (int fch = 0; fch < 2; ++fch) {
#pragma unroll
        for (int i_idx = 0; i_idx < LOAD_INSTS; ++i_idx) {
#pragma unroll
            for (int byte_b = 0; byte_b < 16; ++byte_b) {
                const uint8_t got = v_reg_loaded_byte(v_regs, KV_REG_DWORDS, fch, i_idx, byte_b);
                const uint8_t exp_addr =
                    v_mem_load_expected_byte<SUB_KV, BLOCK_SIZE, NUM_PAIRS, IMM_STRIDE>(
                        v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, block_size, lane,
                        wave, tg_idx, bt_slots, fch, i_idx, byte_b);
                ++local_compared;
                if (got != exp_addr) {
                    ++local_addr_mm;
                }

                int kv_f = 0;
                int d_f = 0;
                v_mem_load_decode_coord_formula<SUB_KV, BLOCK_SIZE>(lane, wave, fch, i_idx, byte_b,
                                                                    IMM_STRIDE, kv_f, d_f);
                if (kv_f < tile_kv && d_f < HEAD_DIM &&
                    got != v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk, stride_kvhead,
                                           tg_idx, block_size, kv_f, d_f)) {
                    ++local_formula_mm;
                }

                const U64 load_addr = v_mem_load_byte_address<SUB_KV, BLOCK_SIZE, NUM_PAIRS,
                                                                IMM_STRIDE>(
                    v_pool, page_ids, valid_blks, stride_blk, stride_kvhead, lane, wave, tg_idx,
                    bt_slots, fch, i_idx, byte_b);
                int kv_a = 0;
                int d_a = 0;
                const bool found =
                    v_mem_load_decode_from_addr<BLOCK_SIZE>(v_pool, page_ids, valid_blks,
                                                            stride_blk, load_addr, kv_a, d_a);
                const bool addr_decode_ok =
                    found && kv_a < tile_kv && d_a < HEAD_DIM &&
                    got == v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk, stride_kvhead,
                                           tg_idx, block_size, kv_a, d_a);
                if (!addr_decode_ok) {
                    ++local_addr_decode_mm;
                    if (first_mismatch_pack == 0xffffffffu) {
                        first_mismatch_pack =
                            v_reg_loc_pack(fch, i_idx, byte_b) ^ v_kv_d_pack(kv_a, d_a);
                    }
                    int kv_b = 0;
                    int d_b = 0;
                    int match_count = 0;
                    if (v_brute_decode_from_hbm<HEAD_DIM, BLOCK_SIZE>(
                            v_pool, page_ids, valid_blks, stride_blk, block_size, tile_kv, got,
                            kv_b, d_b, match_count)) {
                        if (match_count > 1) {
                            ++local_brute_ambig;
                        }
                        if (got != v_hbm_tile_byte(v_pool, page_ids, valid_blks, stride_blk,
                                                   stride_kvhead, tg_idx, block_size, kv_b, d_b)) {
                            ++local_brute_mm;
                        }
                    } else {
                        ++local_brute_mm;
                    }
                }
            }
        }
    }

    smem_stats[threadIdx.x * 8 + 0] = static_cast<float>(local_addr_mm);
    smem_stats[threadIdx.x * 8 + 1] = static_cast<float>(local_formula_mm);
    smem_stats[threadIdx.x * 8 + 2] = static_cast<float>(local_addr_decode_mm);
    smem_stats[threadIdx.x * 8 + 3] = static_cast<float>(local_brute_mm);
    smem_stats[threadIdx.x * 8 + 4] = static_cast<float>(local_brute_ambig);
    smem_stats[threadIdx.x * 8 + 5] = static_cast<float>(local_compared);
    smem_stats[threadIdx.x * 8 + 6] = __int_as_float(first_mismatch_pack);
    smem_stats[threadIdx.x * 8 + 7] = 0.f;
}

// Reduce per-lane bisect stats written to smem[tid*8 + {0..7}].
__device__ __forceinline__ void v_regs_bisect_reduce(const float* smem_lane_stats, float* dbg) {
    if (threadIdx.x == 0) {
        float sums[8] = {};
        uint32_t first_pack = 0xffffffffu;
        for (int i = 0; i < blockDim.x; ++i) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sums[k] += smem_lane_stats[i * 8 + k];
            }
            const uint32_t pack = __float_as_uint(smem_lane_stats[i * 8 + 6]);
            if (first_pack == 0xffffffffu && pack != 0xffffffffu) {
                first_pack = pack;
            }
        }
        if (dbg != nullptr) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                dbg[k] = sums[k];
            }
            dbg[6] = __int_as_float(first_pack);
        }
    }
    __syncthreads();
}

}  // namespace pa_decode
