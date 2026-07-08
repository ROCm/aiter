// Device-side helpers ported from PA_A16W8_Q8_2TG_4W_16mx1_64nx4.sp3
#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace pa_decode {

struct U64 {
    uint32_t lo;
    uint32_t hi;
};

__device__ __forceinline__ U64 u64_from_ptr(const void* p) {
    const auto v = reinterpret_cast<uintptr_t>(p);
    return U64{static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32)};
}

__device__ __forceinline__ U64 u64_add(U64 a, U64 b) {
    U64 r;
    r.lo = a.lo + b.lo;
    const uint32_t carry = (r.lo < a.lo) ? 1u : 0u;
    r.hi = a.hi + b.hi + carry;
    return r;
}

__device__ __forceinline__ U64 u64_add_imm(U64 a, uint32_t imm) {
    return u64_add(a, U64{imm, 0});
}

__device__ __forceinline__ U64 u64_mul_u32(uint32_t a, uint32_t b) {
    const unsigned long long p = static_cast<unsigned long long>(a) * b;
    return U64{static_cast<uint32_t>(p), static_cast<uint32_t>(p >> 32)};
}

__device__ __forceinline__ const uint8_t* u64_to_ptr(U64 a) {
    return reinterpret_cast<const uint8_t*>(static_cast<unsigned long long>(a.hi) << 32 | a.lo);
}

// sp3: K_mem_va_upd / V_mem_va_upd — page_id * stride_blk + (buf_base + lane_base)
__device__ __forceinline__ U64 paged_byte_address(uint32_t page_id,
                                                  uint32_t stride_blk,
                                                  U64 combined_base) {
    return u64_add(u64_mul_u32(page_id, stride_blk), combined_base);
}

__device__ __forceinline__ int lane_id() { return threadIdx.x & 63; }

__device__ __forceinline__ int wave_id() { return threadIdx.x >> 6; }

// sp3 BT_mem_load_addr_gen: per-lane dword index into block table for first prefetch
__device__ __forceinline__ uint32_t block_table_lane_index(int lane, int wave) {
    const int m_id = lane & 3;
    const int q_id = lane & 0xc;
    const int h_id = lane >> 4;

    uint32_t idx = 0;
    if (m_id == 0) {
        idx = static_cast<uint32_t>(q_id + wave);
    } else if (m_id == 1) {
        idx = static_cast<uint32_t>(h_id + q_id);
    }
    return idx;
}

// sp3: v_lshlrev 4 on lane for K, (lane&0xf)<<4 for V — byte offset within page tile
__device__ __forceinline__ uint32_t k_lane_byte_offset(int lane) {
    return static_cast<uint32_t>(lane) << 4;
}

__device__ __forceinline__ uint32_t v_lane_byte_offset(int lane) {
    return static_cast<uint32_t>(lane & 0xf) << 4;
}

// sp3 KVQ_mem_load_addr_gen — byte offset for KQ/VQ buffer_load
__device__ __forceinline__ uint32_t kvq_load_byte_offset(int lane) {
    const int h_id = lane >> 4;
    const int r_id = lane & 0xf;
    const int p_id = r_id >> 3;
    const int q_id = r_id & 3;
    return static_cast<uint32_t>((h_id << 2) + (p_id << 6) + q_id) << 2;
}

// sp3 row_newbcast [0,4,8,12] for K and [1,5,9,13] for V — map pair j -> BT slot
__device__ __forceinline__ uint32_t page_id_for_va_pair(const uint32_t* page_ids,
                                                        int lane,
                                                        int wave,
                                                        int pair_j,
                                                        bool is_v,
                                                        int bt_slots,
                                                        int valid_blks) {
    const int row_base = lane & ~0xf;
    const int ref_lane = row_base + (is_v ? (pair_j * 4 + 1) : (pair_j * 4));
    const uint32_t bt_idx = block_table_lane_index(ref_lane, wave);
    if (bt_idx >= static_cast<uint32_t>(bt_slots) || bt_idx >= static_cast<uint32_t>(valid_blks)) {
        return 0u;
    }
    return page_ids[bt_idx];
}

// sp3 K_mem_va_upd(f_idx) — four 64-bit absolute addresses per lane
template<int NUM_PAIRS>
__device__ __forceinline__ void k_mem_va_upd(U64 out_addrs[NUM_PAIRS],
                                             U64 combined_base,
                                             const uint32_t* page_ids,
                                             int lane,
                                             int wave,
                                             int bt_slots,
                                             int valid_blks,
                                             uint32_t stride_blk) {
#pragma unroll
    for (int j = 0; j < NUM_PAIRS; ++j) {
        const uint32_t pg =
            page_id_for_va_pair(page_ids, lane, wave, j, false, bt_slots, valid_blks);
        out_addrs[j] = paged_byte_address(pg, stride_blk, combined_base);
    }
}

template<int NUM_PAIRS>
__device__ __forceinline__ void v_mem_va_upd(U64 out_addrs[NUM_PAIRS],
                                             U64 combined_base,
                                             const uint32_t* page_ids,
                                             int lane,
                                             int wave,
                                             int bt_slots,
                                             int valid_blks,
                                             uint32_t stride_blk) {
#pragma unroll
    for (int j = 0; j < NUM_PAIRS; ++j) {
        const uint32_t pg = page_id_for_va_pair(page_ids, lane, wave, j, true, bt_slots, valid_blks);
        out_addrs[j] = paged_byte_address(pg, stride_blk, combined_base);
    }
}

// sp3 global_load_dwordx4 — 16-byte global load into 4 dwords (sp3: global_load_dwordx4 ... off)
__device__ __forceinline__ void global_load_dwordx4(uint32_t* dst4, U64 addr) {
    const uint32_t* dw = reinterpret_cast<const uint32_t*>(u64_to_ptr(addr));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst4[i] = dw[i];
    }
}

// sp3 K_mem_load(fch_idx, s, n)
template<int LOAD_INSTS, int REG_DWORDS, int IMM_STRIDE>
__device__ __forceinline__ void k_mem_load(uint32_t* k_regs,
                                           const U64 k_addrs[4],
                                           int fch_idx,
                                           int reg_stride_dwords) {
    const int v_off = fch_idx * reg_stride_dwords;
#pragma unroll
    for (int i_idx = 0; i_idx < LOAD_INSTS; ++i_idx) {
        const int pair = i_idx >> 1;
        const uint32_t imm = (i_idx & 1) ? IMM_STRIDE : 0u;
        U64 load_addr = u64_add_imm(k_addrs[pair], imm);
        global_load_dwordx4(k_regs + v_off + i_idx * 4, load_addr);
    }
}

// sp3 V_mem_load(fch_idx, s, n) — uses (i_idx/4)*IMM and address pair (i_idx&3)
template<int LOAD_INSTS, int REG_DWORDS, int IMM_STRIDE>
__device__ __forceinline__ void v_mem_load(uint32_t* v_regs,
                                           const U64 v_addrs[4],
                                           int fch_idx,
                                           int reg_stride_dwords) {
    const int v_off = fch_idx * reg_stride_dwords;
#pragma unroll
    for (int i_idx = 0; i_idx < LOAD_INSTS; ++i_idx) {
        const int pair = i_idx & 3;
        const uint32_t imm = (i_idx >> 2) * IMM_STRIDE;
        U64 load_addr = u64_add_imm(v_addrs[pair], imm);
        global_load_dwordx4(v_regs + v_off + i_idx * 4, load_addr);
    }
}

// sp3 KQ_mem_va_upd + buffer_load_dword for per-token scale (f_idx = 0 for first tile)
__device__ __forceinline__ float kvq_mem_load_scale(const float* kq_base,
                                                    int tg_idx,
                                                    int lane,
                                                    int wave,
                                                    uint32_t bt_slot,
                                                    uint32_t scale_stride,
                                                    uint32_t kv_nheads) {
    (void)kv_nheads;
    const uint32_t offs = kvq_load_byte_offset(lane);
    const uint32_t byte_off = tg_idx * 64u + bt_slot * scale_stride + offs;
    return kq_base[byte_off / sizeof(float)];
}

__device__ __forceinline__ uint32_t kvq_bt_slot_for_scale(int lane, int wave) {
    // sp3 quad_perm:[0,0,0,0] broadcast of BT[f_idx] within quad
    const int quad_lane = lane & ~3;
    return block_table_lane_index(quad_lane, wave);
}

__device__ __forceinline__ uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// sp3 core_loop BT update: load page ids for KV range [kv_offset, kv_offset+SUB_KV).
template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ void load_block_table_tile_offset(const uint32_t* block_table,
                                                             int batch,
                                                             int max_blks,
                                                             uint32_t kv_seq,
                                                             int kv_offset,
                                                             uint32_t* out_page_ids,
                                                             int& valid_blks) {
    const int bt_dwords = SUB_KV / BLOCK_SIZE;
    const int bt_start = kv_offset / BLOCK_SIZE;
    const int total_blks = static_cast<int>((kv_seq + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int tile_end = static_cast<int>(min_u32(kv_seq, static_cast<uint32_t>(kv_offset + SUB_KV)));
    valid_blks = (tile_end - kv_offset + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const uint32_t* bt_batch = block_table + batch * max_blks;
    for (int i = threadIdx.x; i < bt_dwords; i += blockDim.x) {
        const int global_blk = bt_start + i;
        out_page_ids[i] = (global_blk < total_blks) ? bt_batch[global_blk] : 0u;
    }
    __syncthreads();
}

template<int SUB_KV, int BLOCK_SIZE>
__device__ __forceinline__ void load_block_table_tile(const uint32_t* block_table,
                                                      int batch,
                                                      int max_blks,
                                                      uint32_t kv_seq,
                                                      uint32_t* out_page_ids,
                                                      int& valid_blks) {
    load_block_table_tile_offset<SUB_KV, BLOCK_SIZE>(block_table, batch, max_blks, kv_seq, 0,
                                                     out_page_ids, valid_blks);
}

template<int SUB_Q, int HEAD_DIM, int GQA_RATIO, int Q_LDS_ROWS, int Q_LDS_ROW_ELEMS>
__device__ __forceinline__ void load_q_tile_to_shared(const bf16_t* q_global,
                                                      bf16_t* q_lds,
                                                      int lds_row_stride_elems) {
    static_assert(Q_LDS_ROWS >= SUB_Q, "Q LDS rows must cover MFMA M");
    (void)lds_row_stride_elems;

    for (int row = 0; row < GQA_RATIO; ++row) {
        const bf16_t* src_row = q_global + row * HEAD_DIM;
        bf16_t* dst_row = q_lds + row * Q_LDS_ROW_ELEMS;
        for (int col = threadIdx.x; col < HEAD_DIM; col += blockDim.x) {
            dst_row[col] = src_row[col];
        }
        for (int col = HEAD_DIM + threadIdx.x; col < Q_LDS_ROW_ELEMS; col += blockDim.x) {
            dst_row[col] = bf16_t(0);
        }
    }

    for (int row = GQA_RATIO; row < Q_LDS_ROWS; ++row) {
        bf16_t* dst_row = q_lds + row * Q_LDS_ROW_ELEMS;
        for (int col = threadIdx.x; col < Q_LDS_ROW_ELEMS; col += blockDim.x) {
            dst_row[col] = bf16_t(0);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t checksum_dwords(const uint32_t* data, int n_dwords) {
    uint32_t sum = 0;
#pragma unroll
    for (int i = 0; i < n_dwords; ++i) {
        sum += data[i];
    }
    return sum;
}

}  // namespace pa_decode
