// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "v1_comm.cuh"

template <int32_t kPackedQoLenPerWg_,
          bool kQoSplits_,
          int32_t kUniSeqlenQo_,
          bool kKvSeqlenInLds_,
          bool kIsSparse_ = false>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kPackedQoLenPerWg       = kPackedQoLenPerWg_;
    static constexpr int32_t kPackedQoLenPerWg_log2  = __builtin_ctz(kPackedQoLenPerWg);
    static constexpr bool    kQoSplits               = kQoSplits_;
    // <= -1: read from seqlens_qo_indptr
    // ==  0: read from MlaMetadataV1KernelParameter::uni_seqlen_QO
    // >=  1: read from FlashMlaKernelTrait::kUniSeqlenQo
    static constexpr int32_t kUniSeqlenQo            = kUniSeqlenQo_;
    static constexpr int32_t kFixedOverheadNumBlocks = 16;
    static constexpr int32_t kIsSparse               = kIsSparse_;
    static constexpr bool    kKvSeqlenInLds          = kKvSeqlenInLds_;
};

template <typename Traits>
class QoState
{
public:
    CK_TILE_DEVICE explicit QoState(
        const int32_t uni_seqlen_qo,
        const int32_t* p_lds_seqlens_qo,
        const int32_t* p_seqlens_qo_indptr) :
        uni_seqlen_qo_(uni_seqlen_qo),
        p_lds_seqlens_qo_(p_lds_seqlens_qo),
        p_seqlens_qo_indptr_(p_seqlens_qo_indptr)
    { }

    CK_TILE_DEVICE int32_t get_seqlen(
        const int32_t batch_idx)
    {
        if constexpr (Traits::kUniSeqlenQo == 0)
        {
            return uni_seqlen_qo_;
        }
        else if constexpr (Traits::kUniSeqlenQo <= -1)
        {
            return p_lds_seqlens_qo_[batch_idx];
        }
        else
        {
            return Traits::kUniSeqlenQo;
        }
    }

    CK_TILE_DEVICE int32_t get_begin(
        const int32_t batch_idx)
    {
        if constexpr (Traits::kUniSeqlenQo == 0)
        {
            return uni_seqlen_qo_ * batch_idx;
        }
        else if constexpr (Traits::kUniSeqlenQo <= -1)
        {
            return p_seqlens_qo_indptr_[batch_idx];
        }
        else
        {
            return Traits::kUniSeqlenQo * batch_idx;
        }
    }

    CK_TILE_DEVICE int32_t get_end(
        const int32_t batch_idx)
    {
        if constexpr (Traits::kUniSeqlenQo == 0)
        {
            return uni_seqlen_qo_ * (batch_idx + 1);
        }
        else if constexpr (Traits::kUniSeqlenQo <= -1)
        {
            return p_seqlens_qo_indptr_[batch_idx + 1];
        }
        else
        {
            return Traits::kUniSeqlenQo * (batch_idx + 1);
        }
    }

private:
    const int32_t uni_seqlen_qo_;
    const int32_t* const p_lds_seqlens_qo_;
    const int32_t* const p_seqlens_qo_indptr_;
};

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1_2(
    MlaMetadataV1KernelParameter params)
{
    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_seqlens_kv = p_lds_seqlens_qo + params.num_batches;

    QoState<Traits> qo_state(params.uni_seqlen_qo, p_lds_seqlens_qo, params.p_seqlens_qo_indptr);

    const int32_t lane_idx = ck_tile::get_lane_id();

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    int32_t sum_blocks = 0;
    for (int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        int bid_ori = [&]() {
            if constexpr (!Traits::kIsSparse)
            {
                return bid;
            }
            else
            {
                return bid / params.ori_seqlen_qo;
            }
        }();
        int32_t kv_end = params.p_seqlens_kv_indptr[bid_ori + 1];
        int32_t seqlen_kv = kv_end - params.p_seqlens_kv_indptr[bid_ori];
        seqlen_kv = Traits::kIsSparse ? min(seqlen_kv, params.topk) : seqlen_kv;
        if constexpr (Traits::kKvSeqlenInLds)
        {
            p_lds_seqlens_kv[bid] = seqlen_kv;
        }

        const int32_t num_blocks =
            integer_divide_ceil_power2(seqlen_kv, params.kv_granularity, params.kv_granularity_log2);
        sum_blocks += num_blocks;

        if constexpr (Traits::kUniSeqlenQo == -1)
        {
            p_lds_seqlens_qo[bid] = params.p_seqlens_qo_indptr[bid + 1] - params.p_seqlens_qo_indptr[bid];
        }
    }

    sum_blocks = aiter::warpReduce<aiter::AddFunctor, decltype(sum_blocks), ck_tile::get_warp_size()>(sum_blocks);
    sum_blocks += params.num_batches * Traits::kFixedOverheadNumBlocks;

    if (lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
    }

    // expected payload handled by each cu part.
    const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, params.num_cu) +
                            Traits::kFixedOverheadNumBlocks;

    int32_t curr_batch = 0;         // batch ID of the batch which is under review
    int32_t curr_kv_block = 0;      // #blocks handled by previous cu part(s)
    int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch

    int32_t curr_kv_begin  = 0;
    int32_t curr_kv_end    = Traits::kKvSeqlenInLds ? p_lds_seqlens_kv[0] : params.p_seqlens_kv_indptr[1];
    int32_t curr_kv_seqlen = curr_kv_end - curr_kv_begin;

    int32_t num_works = 0;
    int32_t partial_idx = 0;
    int32_t tot_qo_tiles = 0;
    int32_t last_reduce_indptr = 0;

    for (int32_t cid = 0; cid < params.num_cu; ++cid)
    {
        int32_t remain_payload = payload;
        while (curr_batch < params.num_batches)
        {
            const int32_t packed_qo_len = qo_state.get_seqlen(curr_batch) * params.num_heads;
            const int32_t num_qo_tiles = Traits::kQoSplits ?
                integer_divide_ceil_power2(packed_qo_len, Traits::kPackedQoLenPerWg, Traits::kPackedQoLenPerWg_log2) :
                1;
            const int32_t qo_tile_size = ck_tile::integer_divide_ceil(qo_state.get_seqlen(curr_batch), num_qo_tiles);
            const int32_t num_kv_blocks =
                integer_divide_ceil_power2(curr_kv_seqlen, params.kv_granularity, params.kv_granularity_log2);
            const int32_t remain_kv_blocks = num_kv_blocks - curr_kv_block;

            // If current cu part is able to handle this batch of seqences
            if (remain_payload >= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks))
            {
                const int32_t num_splits = curr_n_split_idx + 1;

                auto fill_work_info = [&](const int32_t qo_tile_idx, const int32_t split_idx)
                {
                    const int32_t global_qo_tile_idx = tot_qo_tiles + qo_tile_idx;

                    MlaWorkInfo work_info{};
                    work_info.batch_idx = curr_batch;
                    work_info.qo_start = qo_state.get_begin(curr_batch) + qo_tile_idx * qo_tile_size;
                    work_info.qo_end = ck_tile::min(work_info.qo_start + qo_tile_size, qo_state.get_end(curr_batch));
                    work_info.kv_start = curr_kv_begin + (curr_kv_block * params.kv_granularity);
                    work_info.kv_end =
                        ck_tile::min(work_info.kv_start + (remain_kv_blocks * params.kv_granularity),
                                     curr_kv_end - (num_qo_tiles - 1 - qo_tile_idx));
                    work_info.kv_offset = curr_kv_end - work_info.kv_end;

                    // split related info
                    if (curr_n_split_idx > 0)
                    {
                        // set work info
                        work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;

                        // set reduce info
                        params.p_reduce_indptr[global_qo_tile_idx + 1] =
                            last_reduce_indptr + (qo_tile_idx + 1) * num_splits;
                        params.p_reduce_final_map[global_qo_tile_idx * 2] = work_info.qo_start;
                        params.p_reduce_final_map[global_qo_tile_idx * 2 + 1] = work_info.qo_end;
                        params.p_reduce_partial_map[last_reduce_indptr + qo_tile_idx * num_splits + split_idx] =
                            partial_idx - (curr_n_split_idx - split_idx) * qo_tile_size * num_qo_tiles;
                    }
                    else
                    {
                        work_info.partial_qo_loc = -1;
                        params.p_reduce_indptr[global_qo_tile_idx + 1] = last_reduce_indptr;
                        // params.p_reduce_final_map[global_qo_tile_idx * 2] = -1;
                        // params.p_reduce_final_map[global_qo_tile_idx * 2 + 1] = -2;
                    }

                    p_work_info_set[num_works + qo_tile_idx] = work_info;
                };

                // record a work in work_info_set
                if constexpr (Traits::kQoSplits)
                {
                    if (curr_n_split_idx > 0)
                    {
                        for (int32_t idx = lane_idx; idx < num_splits * num_qo_tiles; idx += ck_tile::get_warp_size())
                        {
                            const int32_t qo_tile_idx = idx % num_qo_tiles;
                            const int32_t split_idx   = idx / num_qo_tiles;
                            fill_work_info(qo_tile_idx, split_idx);
                        }

                        partial_idx += num_qo_tiles * qo_tile_size;
                        last_reduce_indptr += num_qo_tiles * num_splits;
                    }
                    else
                    {
                        for (int32_t idx = lane_idx; idx < num_qo_tiles; idx += ck_tile::get_warp_size())
                        {
                            fill_work_info(idx, 0);
                        }
                    }
                }
                else
                {
                    if (curr_n_split_idx > 0)
                    {
                        for (int32_t idx = lane_idx; idx < num_splits; idx += ck_tile::get_warp_size())
                        {
                            fill_work_info(0, idx);
                        }

                        partial_idx += qo_tile_size;
                        last_reduce_indptr += num_splits;
                    }
                    else
                    {
                        fill_work_info(0, 0);
                    }
                }

                tot_qo_tiles += num_qo_tiles;
                num_works += num_qo_tiles;

                // update state
                remain_payload -= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks);
                ++curr_batch;
                if (curr_batch < params.num_batches)
                {
                    curr_kv_block = 0;
                    curr_n_split_idx = 0;

                    if constexpr (Traits::kKvSeqlenInLds)
                    {
                        curr_kv_seqlen = p_lds_seqlens_kv[curr_batch];
                    }
                    else
                    {
                        const int32_t bid_ori = curr_batch / params.ori_seqlen_qo;
                        curr_kv_seqlen = params.p_seqlens_kv_indptr[bid_ori + 1] - params.p_seqlens_kv_indptr[bid_ori];
                        curr_kv_seqlen = Traits::kIsSparse ? min(curr_kv_seqlen, params.topk) : curr_kv_seqlen;
                    }
                    curr_kv_begin = Traits::kIsSparse ? curr_batch * params.topk : curr_kv_end;
                    curr_kv_end = curr_kv_begin + curr_kv_seqlen;
                }
            }
            else
            {
                if (remain_payload > Traits::kFixedOverheadNumBlocks)
                {
                    const int32_t consuming_blks = remain_payload - Traits::kFixedOverheadNumBlocks;

                    auto fill_work_info = [&](const int32_t qo_tile_idx)
                    {
                        MlaWorkInfo work_info{};
                        work_info.batch_idx = curr_batch;
                        work_info.qo_start = qo_state.get_begin(curr_batch) + qo_tile_idx * qo_tile_size;
                        work_info.qo_end =
                            ck_tile::min(work_info.qo_start + qo_tile_size, qo_state.get_end(curr_batch));
                        work_info.kv_start = curr_kv_begin + (curr_kv_block * params.kv_granularity);
                        work_info.kv_end = 
                            ck_tile::min(work_info.kv_start + (consuming_blks * params.kv_granularity),
                            curr_kv_end - (num_qo_tiles - 1 - qo_tile_idx));
                        work_info.kv_offset = curr_kv_end - work_info.kv_end;
                        work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;
                        p_work_info_set[num_works + qo_tile_idx] = work_info;
                    };

                    // record a work in work_info_set
                    if constexpr (Traits::kQoSplits)
                    {
                        for (int32_t qo_tile_idx = lane_idx;
                            qo_tile_idx < num_qo_tiles;
                            qo_tile_idx += ck_tile::get_warp_size())
                        {
                            fill_work_info(qo_tile_idx);
                        }
                    }
                    else
                    {
                        fill_work_info(0);
                    }

                    partial_idx += num_qo_tiles * qo_tile_size;
                    num_works += num_qo_tiles;

                    // update state
                    curr_kv_block += consuming_blks;
                    ++curr_n_split_idx;
                }
                break;
            }
        }

        params.p_work_indptr[cid + 1] = num_works;
    }

    for (int32_t i = params.num_batches + lane_idx; i < params.reduce_indptr_size; i += ck_tile::get_warp_size())
    {
        params.p_reduce_indptr[i] = last_reduce_indptr;
    }
}

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1_2_no_split(
    MlaMetadataV1KernelParameter params)
{
    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_seqlens_kv = p_lds_seqlens_qo;   // p_lds_seqlens_qo should be params.num_batches in length but it is
                                                    // not used here.

    const int32_t lane_idx = ck_tile::get_lane_id();

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    int32_t sum_blocks = 0;

    for (int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        int bid_ori = [&]() {
            if constexpr (Traits::kIsSparse || Traits::kQoSplits)
            {
                return bid / params.ori_seqlen_qo;
            }
            else
            {
                return bid;
            }
        }();
        int32_t kv_end = params.p_seqlens_kv_indptr[bid_ori + 1];
        int32_t seqlen_kv = kv_end - params.p_seqlens_kv_indptr[bid_ori];
        seqlen_kv = Traits::kIsSparse ? min(seqlen_kv, params.topk) : seqlen_kv;
        if constexpr (Traits::kKvSeqlenInLds)
        {
            p_lds_seqlens_kv[bid] = seqlen_kv;
        }

        const int32_t num_blocks = int32_t((seqlen_kv + 15) >> 4);
        sum_blocks += num_blocks;
    }

    sum_blocks = aiter::warpReduce<aiter::AddFunctor, decltype(sum_blocks), ck_tile::get_warp_size()>(sum_blocks);
    sum_blocks += params.num_batches * Traits::kFixedOverheadNumBlocks;

    if (lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_metadata_ptrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_reduce_partial_map[0] = -1;
    }
    else
    {
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
		params.p_reduce_partial_map[1] = Traits::kUniSeqlenQo;
    }

    // expected payload handled by each cu part.
    const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, params.num_cu) +
                            Traits::kFixedOverheadNumBlocks;

    int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch


    int32_t curr_batch = 0;         // batch ID of the batch which is under review
    int32_t curr_kv_block = 0;      // #blocks handled by previous cu part(s)

    int32_t curr_kv_begin  = 0;
    int32_t curr_kv_end    = Traits::kKvSeqlenInLds ? p_lds_seqlens_kv[0] :
                             Traits::kIsSparse ? min(params.p_seqlens_kv_indptr[1], params.topk) :
                             params.p_seqlens_kv_indptr[1];
    int32_t curr_kv_seqlen = curr_kv_end - curr_kv_begin;

    int32_t num_works = 0;
    int32_t num_partial = 0;
    int32_t partial_idx = 0;
    int32_t last_reduce_indptr = 0;

    for (int32_t cid = 0; cid < params.num_cu; ++cid)
    {
        int32_t remain_payload = payload;
        while (curr_batch < params.num_batches)
        {
            // const int32_t num_kv_blocks = ck_tile::integer_divide_ceil(curr_kv_seqlen, params.kv_granularity);
            const int32_t num_kv_blocks = int32_t((curr_kv_seqlen + 15) >> 4);
            const int32_t remain_kv_blocks = num_kv_blocks - curr_kv_block;
            MlaWorkInfo work_info{};

            // If current cu part is able to handle this batch of seqences
            if (remain_payload >= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks))
            {
                const int32_t num_splits = curr_n_split_idx + 1;
                int32_t reduce_indptr = last_reduce_indptr;

                work_info.batch_idx = curr_batch;
                work_info.qo_start = curr_batch * Traits::kUniSeqlenQo;
                work_info.qo_end = work_info.qo_start + Traits::kUniSeqlenQo;
                work_info.kv_start = curr_kv_begin + curr_kv_block * params.kv_granularity;
                work_info.kv_end =
                    ck_tile::min(work_info.kv_start + remain_kv_blocks * params.kv_granularity, curr_kv_end);
                work_info.kv_offset = curr_kv_end - work_info.kv_end;

                work_info.partial_qo_loc = -1;

                if (curr_n_split_idx > 0)
                {
                    // set work info
                    work_info.partial_qo_loc = partial_idx;
                    // set reduce info
                    reduce_indptr = last_reduce_indptr + num_splits;

                    partial_idx += Traits::kUniSeqlenQo;
                    last_reduce_indptr += num_splits;
                }

                params.p_reduce_indptr[curr_batch + 1] = reduce_indptr;
                p_work_info_set[num_works] = work_info;

                ++num_works;

                // update state
                remain_payload -= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks);
                ++curr_batch;

                if (curr_batch < params.num_batches)
                {
                    curr_kv_block = 0;
                    curr_n_split_idx = 0;
                    curr_kv_begin = Traits::kIsSparse ? curr_batch * params.topk : curr_kv_end;
                    if constexpr (Traits::kKvSeqlenInLds)
                    {
                        curr_kv_seqlen = p_lds_seqlens_kv[curr_batch];
                    }
                    else
                    {
                        const int32_t bid_ori = curr_batch / params.ori_seqlen_qo;
                        curr_kv_seqlen = params.p_seqlens_kv_indptr[bid_ori + 1] - params.p_seqlens_kv_indptr[bid_ori];
                        curr_kv_seqlen = Traits::kIsSparse ? min(curr_kv_seqlen, params.topk) : curr_kv_seqlen;
                    }
                    curr_kv_end = curr_kv_begin + curr_kv_seqlen;
                }
            }
            else
            {
                if (remain_payload > Traits::kFixedOverheadNumBlocks)
                {
                    const int32_t consuming_blks = remain_payload - Traits::kFixedOverheadNumBlocks;

                    // record a work in work_info_set
                    work_info.batch_idx = curr_batch;
                    work_info.qo_start = curr_batch * Traits::kUniSeqlenQo;
                    work_info.qo_end = work_info.qo_start + Traits::kUniSeqlenQo;
                    work_info.kv_start = curr_kv_begin + curr_kv_block * params.kv_granularity;
                    work_info.kv_end = work_info.kv_start + consuming_blks * params.kv_granularity; // TODO: we need to take casual mask into consideration
                    work_info.kv_offset = curr_kv_end - work_info.kv_end;
                    work_info.partial_qo_loc = partial_idx;
                    p_work_info_set[num_works] = work_info;

                    partial_idx += Traits::kUniSeqlenQo;
                    ++num_works;

                    // update state
                    curr_kv_block += consuming_blks;
                    ++curr_n_split_idx;
                }
                break;
            }
        }

        params.p_work_indptr[cid + 1] = num_works;
    }

    for (int32_t i = params.num_batches + lane_idx; i < params.reduce_indptr_size; i += blockDim.x)
    {
        params.p_reduce_indptr[i] = last_reduce_indptr;
    }
}

#define MLA_UNI_SEQLEN_QO_CASE(UNI_SEQLEN_QO, ...)                        \
    case UNI_SEQLEN_QO: \
    {   \
        constexpr int32_t kUniSeqlenQo = UNI_SEQLEN_QO; \
        if (num_lds_elem >= params.num_batches) \
        { \
            constexpr bool kKvSeqlenInLds = true; \
            using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
            __VA_ARGS__; \
        } \
        else \
        { \
            constexpr bool kKvSeqlenInLds = false; \
            using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
            __VA_ARGS__; \
        } \
        break; \
    }

#define MLA_UNI_SEQLEN_DISPATCHER(...)                        \
    const int32_t num_lds_elem = dev_prop.maxSharedMemoryPerMultiProcessor / sizeof(int32_t); \
    switch (params.uni_seqlen_qo) \
    { \
        MLA_UNI_SEQLEN_QO_CASE(1, __VA_ARGS__); \
        MLA_UNI_SEQLEN_QO_CASE(2, __VA_ARGS__); \
        MLA_UNI_SEQLEN_QO_CASE(3, __VA_ARGS__); \
        MLA_UNI_SEQLEN_QO_CASE(4, __VA_ARGS__); \
        default: \
        { \
            if (params.uni_seqlen_qo > 0) \
            { \
                constexpr int32_t kUniSeqlenQo = 0; \
                if (num_lds_elem >= params.num_batches) \
                { \
                    constexpr bool kKvSeqlenInLds = true; \
                    using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
                    __VA_ARGS__; \
                } \
                else \
                { \
                    constexpr bool kKvSeqlenInLds = false; \
                    using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
                    __VA_ARGS__; \
                } \
            } \
            else \
            { \
                constexpr int32_t kUniSeqlenQo = -1;  \
                if (num_lds_elem >= params.num_batches) \
                { \
                    constexpr bool kKvSeqlenInLds = true; \
                    using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
                    __VA_ARGS__; \
                } \
                else \
                { \
                    constexpr bool kKvSeqlenInLds = false; \
                    using Traits = FlashMlaKernelTrait<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kKvSeqlenInLds, kIsSparse>; \
                    __VA_ARGS__; \
                } \
            } \
            break; \
        } \
    }

void get_mla_metadata_v1_2_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const int32_t        kv_granularity,
    const int32_t        max_seqlen_qo,
    const int32_t        uni_seqlen_qo,
    const int32_t        topk,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    constexpr int32_t kPackedQoLenPerWg = 128;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);

    const int32_t num_clusters = dev_prop.multiProcessorCount / num_heads_k;

    MlaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs = work_metadata_ptrs.data_ptr<uint64_t>();
    params.p_work_indptr        = work_indptr.data_ptr<int32_t>();
    params.p_work_info_set_raw  = work_info_set.data_ptr<int32_t>();
    params.p_reduce_indptr      = reduce_indptr.data_ptr<int32_t>();
    params.p_reduce_final_map   = reduce_final_map.data_ptr<int32_t>();
    params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
    params.p_seqlens_qo_indptr  = seqlens_qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_kv_indptr  = seqlens_kv_indptr.data_ptr<int32_t>();
    params.num_batches          = seqlens_kv_indptr.size(0) - 1;
    params.num_heads            = num_heads_k * num_heads_per_head_k;
    params.num_cu               = num_clusters;
    params.reduce_indptr_size   = reduce_indptr.size(0);
    params.kv_granularity       = kv_granularity;
    params.kv_granularity_log2  = __builtin_ctz(kv_granularity);
    params.uni_seqlen_qo        = uni_seqlen_qo;
    params.is_causal            = is_causal;
    params.topk                 = topk;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity

    if ((max_seqlen_qo > 0) && (max_seqlen_qo * num_heads_per_head_k <= kPackedQoLenPerWg))
    {
        constexpr bool kQoSplits = false;
        if (topk == -1)
        {
            constexpr bool kIsSparse = false;
            params.ori_seqlen_qo = 1;
            MLA_UNI_SEQLEN_DISPATCHER(
                kn_get_mla_metadata_v1_2_no_split<Traits>
                    <<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params)
            );
        }
        else
        {
            params.num_batches = uni_seqlen_qo * params.num_batches;
            params.uni_seqlen_qo = 1;
            params.ori_seqlen_qo = uni_seqlen_qo;
            constexpr bool kIsSparse = true;
            MLA_UNI_SEQLEN_DISPATCHER(
                kn_get_mla_metadata_v1_2_no_split<Traits>
                    <<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params)
            );
        }
    }
    else
    {
        params.num_batches = uni_seqlen_qo * params.num_batches;
        params.uni_seqlen_qo = 1;
        params.ori_seqlen_qo = uni_seqlen_qo;
        constexpr bool kQoSplits = true;
        if (topk == -1)
        {
            constexpr bool kIsSparse = false;

            MLA_UNI_SEQLEN_DISPATCHER(
                kn_get_mla_metadata_v1_2_no_split<Traits>
                    <<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params)
            );
        }
        else
        {
            constexpr bool kIsSparse = true;
            MLA_UNI_SEQLEN_DISPATCHER(
                kn_get_mla_metadata_v1_2_no_split<Traits>
                    <<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params)
            );
        }
    }
}
