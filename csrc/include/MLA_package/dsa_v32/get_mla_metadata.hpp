#pragma once

#include <opus/opus.hpp>
#include "defs.h"

namespace opus_mla_decode_metadata {

__device__ inline int effective_splits(const opus_mla_decode_metadata_kargs& kargs, int sum_blocks) {
    if (!kargs.auto_split) return kargs.num_splits;
    const float work = float(opus::max(1, sum_blocks)) * float(opus::max(1, kargs.num_splits));
    const int eff = int(lrintf(MLA_DECODE_SPLIT_COEF * sqrtf(work)));
    return opus::max(1, opus::min(eff, kargs.num_splits));
}

}

__global__ void get_mla_metadata_kernel(opus_mla_decode_metadata_kargs kargs) {
    using namespace opus_mla_decode_metadata;
    constexpr int WARP = 64;

    const int B = kargs.B;
    const int H = kargs.H;
    const int gran = kargs.kv_granularity;
    const int overhead = kargs.fixed_overhead;
    const int lane = opus::thread_id_x();

    int sum_blocks = 0;
    for (int b = lane; b < B; b += WARP) {
        const int seqlen_qo = kargs.qo_indptr[b + 1] - kargs.qo_indptr[b];
        const int seqlen_kv = kargs.kv_indptr[b + 1] - kargs.kv_indptr[b];
        sum_blocks += (ceil_div(seqlen_kv, gran) + overhead) * mla_decode_num_qo_tiles(seqlen_qo, H);
    }
    #pragma unroll
    for (int offset = WARP / 2; offset >= 1; offset /= 2)
        sum_blocks += __builtin_amdgcn_ds_bpermute((lane ^ offset) << 2, sum_blocks);

    const int payload = ceil_div(sum_blocks, effective_splits(kargs, sum_blocks)) + overhead;

    __shared__ int tail[2];

    if (lane == 0) {
        kargs.work_indptr[0] = 0;
        kargs.reduce_indptr[0] = 0;

        int curr_batch = 0, curr_qo_tile_idx = 0, curr_kv_block = 0, curr_n_split_idx = 0;
        bool cur_tail_done = false;
        int num_works = 0, partial_idx = 0, tot_qo_tiles = 0, last_reduce_indptr = 0;

        int curr_kv_begin = (B > 0) ? kargs.kv_indptr[0] : 0;
        int curr_kv_end   = (B > 0) ? kargs.kv_indptr[1] : 0;

        for (int cid = 0; cid < kargs.num_cu; ++cid) {
            int remain_payload = payload;

            while (curr_batch < B) {
                const int qo_begin = kargs.qo_indptr[curr_batch];
                const int qo_limit = kargs.qo_indptr[curr_batch + 1];
                const int num_qo_tiles = mla_decode_num_qo_tiles(qo_limit - qo_begin, H);
                const int qo_tile_size = ceil_div(qo_limit - qo_begin, num_qo_tiles);

                const int num_kv_blocks = ceil_div(curr_kv_end - curr_kv_begin, gran);
                const int remain_kv_blocks = num_kv_blocks - curr_kv_block;

                const int batch_tail = kargs.is_causal
                                     ? opus::max(num_qo_tiles - 1 - curr_qo_tile_idx, 0) : 0;
                const int qo_start = qo_begin + curr_qo_tile_idx * qo_tile_size;
                const int qo_end   = opus::min(qo_start + qo_tile_size, qo_limit);
                const int kv_start = curr_kv_begin + curr_kv_block * gran;

                if (remain_payload >= remain_kv_blocks + overhead || cur_tail_done) {
                    const int num_splits = curr_n_split_idx + 1;

                    int kv_end = opus::min(kv_start + remain_kv_blocks * gran, curr_kv_end - batch_tail);
                    if ((curr_kv_end - kv_end < kargs.tail_done_threshold && curr_kv_end - kv_end > 0)
                        || cur_tail_done)
                        kv_end = curr_kv_end - batch_tail;

                    opus_mla_decode_work_info work_info{};
                    work_info.batch_idx = curr_batch;
                    work_info.partial_slot = (curr_n_split_idx > 0) ? partial_idx : -1;
                    work_info.qo_start = qo_start;
                    work_info.qo_end = qo_end;
                    work_info.kv_start = kv_start;
                    work_info.kv_end = kv_end;
                    work_info.kv_offset = curr_kv_end - kv_end;
                    kargs.work_info_set[num_works] = work_info;

                    if (curr_n_split_idx > 0) {
                        kargs.reduce_indptr[tot_qo_tiles + 1] = last_reduce_indptr + num_splits;
                        kargs.reduce_final_map[tot_qo_tiles * 2] = qo_start;
                        kargs.reduce_final_map[tot_qo_tiles * 2 + 1] = qo_end;
                        for (int s = 0; s < num_splits; ++s)
                            kargs.reduce_partial_map[last_reduce_indptr + s] =
                                partial_idx - (curr_n_split_idx - s) * qo_tile_size;
                        partial_idx += qo_tile_size;
                        last_reduce_indptr += num_splits;
                    } else {
                        kargs.reduce_indptr[tot_qo_tiles + 1] = last_reduce_indptr;
                    }

                    tot_qo_tiles += 1;
                    num_works += 1;
                    remain_payload -= remain_kv_blocks + overhead;

                    curr_qo_tile_idx = (curr_qo_tile_idx == num_qo_tiles - 1) ? 0 : curr_qo_tile_idx + 1;
                    if (curr_qo_tile_idx == 0) {
                        ++curr_batch;
                        if (curr_batch < B) {
                            curr_kv_begin = curr_kv_end;
                            curr_kv_end = kargs.kv_indptr[curr_batch + 1];
                        }
                    }
                    curr_kv_block = 0;
                    curr_n_split_idx = 0;
                    cur_tail_done = false;
                } else {
                    if (remain_payload > overhead) {
                        const int consuming_blks = remain_payload - overhead;

                        int kv_end = opus::min(kv_start + consuming_blks * gran, curr_kv_end - batch_tail);
                        if (curr_kv_end - kv_end < kargs.tail_done_threshold) {
                            cur_tail_done = true;
                            kv_end = curr_kv_end - batch_tail;
                        }

                        if (!cur_tail_done) {
                            opus_mla_decode_work_info work_info{};
                            work_info.batch_idx = curr_batch;
                            work_info.partial_slot = partial_idx;
                            work_info.qo_start = qo_start;
                            work_info.qo_end = qo_end;
                            work_info.kv_start = kv_start;
                            work_info.kv_end = kv_end;
                            work_info.kv_offset = curr_kv_end - kv_end;
                            kargs.work_info_set[num_works] = work_info;

                            partial_idx += qo_tile_size;
                            num_works += 1;
                            curr_kv_block += consuming_blks;
                            ++curr_n_split_idx;
                        }
                    }
                    if (!cur_tail_done) break;
                }
            }

            kargs.work_indptr[cid + 1] = num_works;
        }

        tail[0] = tot_qo_tiles;
        tail[1] = last_reduce_indptr;
    }

    __builtin_amdgcn_s_barrier();

    for (int i = tail[0] + lane; i < kargs.reduce_indptr_size; i += WARP)
        kargs.reduce_indptr[i] = tail[1];
}
