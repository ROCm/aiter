// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#define PRINT_DBG 0

#include <torch/extension.h>

template <typename T>
inline T pack_dword(const T low_part, const T high_part)
{
    T dw = (high_part << 16) | (low_part & 0xFFFF);
    return dw;
}

template <typename T>
inline std::tuple<T, T> unpack_dword(const T dw)
{
    T high_part = (dw >> 16) & 0xFFFF;
    T low_part  = dw & 0xFFFF;
    return std::make_tuple(low_part, high_part);
}

union WorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_o_loc;
        int32_t qo_start;
        int32_t qo_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t q_head_range;

// #if PRINT_DBG
//         friend std::ostream& operator<<(std::ostream& os, const WorkInfo& work)
//         {
//             auto q_heads = unpack_dword(work.q_head_range);
//             os << std::setw(10) << work.batch_idx << "," << std::setw(10) << work.partial_o_loc << ","
//             << std::setw(10) << work.qo_start << "," << std::setw(10) << work.qo_end << ","
//             << std::setw(10) << work.kv_start << "," << std::setw(10) << work.kv_end << ","
//             << std::setw(10) << work.kv_offset << "," << std::setw(10) << work.q_head_range << "["
//             << std::get<0>(q_heads) << "," << std::get<1>(q_heads) << ")";
//             return os;
//         }
// #endif
    };
    uint32_t u32All[8];
};
constexpr size_t kSizeWorkInfoInDw = sizeof(WorkInfo) / sizeof(uint32_t);
static_assert(kSizeWorkInfoInDw == 8);


union FinalLoc
{
    struct
    {
        int32_t qo_start;
        int32_t qo_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeFinalLocInDw = sizeof(FinalLoc) / sizeof(uint32_t);
static_assert(kSizeFinalLocInDw == 2);


struct QTile
{
    int32_t batch_idx;
    int32_t qo_start; // global
    int32_t qo_end;   // global
    int32_t num_blocks;
    int32_t effective_kv_length;

#if PRINT_DBG
    friend std::ostream& operator<<(std::ostream& os, const QTile& qtile)
    {
        os << std::setw(10) << qtile.batch_idx << "," << std::setw(10) << qtile.qo_start << ","
           << std::setw(10) << qtile.qo_end << "," << std::setw(10) << qtile.num_blocks << ","
           << std::setw(10) << qtile.effective_kv_length;
        return os;
    }
#endif
};


void get_ps_metadata_v1(const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
                        const torch::Tensor& pages_kv_indptr,       // [batch size + 1]
                        const torch::Tensor& context_lens,          // [batch size]
                        const int32_t        gqa_ratio,
                        const int32_t        num_heads_k,
                        torch::Tensor&       work_metadata_ptrs,
                        torch::Tensor&       work_indptr,
                        torch::Tensor&       work_info,
                        torch::Tensor&       reduce_indptr,
                        torch::Tensor&       reduce_final_map,
                        torch::Tensor&       reduce_partial_map,
                        const int32_t        qhead_granularity,
                        const int32_t        qlen_granularity,
                        const int32_t        kvlen_granlarity,
                        const int32_t        block_size,
                        const bool           is_causal);


// DEBUG
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    std::ios_base::fmtflags old_flags = os.flags();
    os << "(" << std::dec << vec.size() << ") [";
    for(size_t i = 0; i < vec.size(); ++i)
    {
        os << std::dec << vec[i];
        if(i < vec.size() - 1)
            os << ",";
    }
    os << "]";
    // os.flags(old_flags);
    return os;
}


// void print_metadata(std::vector<int32_t>& work_indptr, std::vector<WorkInfo>& work_info)
// {
//     std::cout << "\n=== PS Metadata ===" << std::endl;
//     const int32_t available_tgs = work_indptr.size() - 1;
//     const int32_t actual_works = work_indptr.back();
//     std::cout << "Number of available TGs: " << available_tgs << std::endl
//               << "Number of actual work items: " << actual_works << std::endl;

//     std::cout << "work_indptr:" << work_indptr << std::endl;

//     std::cout << std::setw(12) << "" << std::setw(11) << "batch_idx  " << std::setw(11)
//               << "partial_loc" << std::setw(11) << "[qo_start" << std::setw(11) << "qo_end)"
//               << std::setw(11) << "[kv_start" << std::setw(11) << "kv_end)" << std::setw(11)
//               << "kv_offset" << std::setw(16) << "[q_head_range)" << std::endl;
//     std::cout << "work_info: (" << work_info.size() << ") [" << std::endl;
//     int32_t tg_idx    = 0;
//     int32_t kv_blocks = 0;
//     for(int32_t i = 0; i < actual_works; ++i)
//     {
//         auto work = work_info[i];
//         kv_blocks += work.kv_end - work.kv_start;
//         std::cout << "work[" << std::setw(3) << i << "]: " << work << std::endl;
//         if(i == work_indptr[tg_idx + 1] - 1)
//         {
//             std::cout << "tg:" << std::setw(2) << tg_idx << ","
//                       << " blk:" << std::setw(4) << kv_blocks << " " << std::string(90, '-')
//                       << std::endl;
//             tg_idx++;
//             kv_blocks = 0;
//         }
//     }
//     std::cout << "]" << std::endl;
// }


// void print_reduce_info(std::vector<int32_t>& reduce_indptr,
//                        std::vector<std::vector<int32_t>>& reduce_final_map,
//                        std::vector<int32_t>& reduce_partial_map)
// {
//     std::cout << "\n=== PS Reduce Info ===" << std::endl;
//     std::cout << "reduce_indptr:" << reduce_indptr << std::endl;

//     std::cout << std::setw(22) << "" << std::setw(13) << "[final_o_start  " << std::setw(13)
//               << "final_o_end)" << std::endl;
//     std::cout << "reduce_final_map: (" << reduce_final_map.size() << ") [" << std::endl;
//     for(size_t i = 0; i < reduce_final_map.size(); ++i)
//     {
//         auto final_loc = reduce_final_map[i];
//         std::cout << "reduce_final_map[" << std::setw(3) << i << "]: " << std::setw(11)
//                   << final_loc.qo_start << "," << std::setw(11) << final_loc.qo_end << std::endl;
//     }
//     std::cout << "]" << std::endl;

//     std::cout << "reduce_partial_map: " << reduce_partial_map << std::endl;
// }
