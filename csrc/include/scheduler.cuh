#include "exception.h"

using IdType = int32_t;

struct MLAPlanInfo {
    int64_t num_blks_x;
    int64_t num_blks_y;
    int64_t q_indptr_offset;
    int64_t kv_indptr_offset;
    int64_t partial_indptr_offset;
    int64_t merge_packed_offset_start_offset;
    int64_t merge_packed_offset_end_offset;
    int64_t merge_indptr_offset;
    int64_t q_len_offset;
    int64_t kv_len_offset;
    int64_t q_start_offset;
    int64_t kv_start_offset;
    int64_t kv_end_offset;
    int64_t work_indptr_offset;
    int64_t partial_o_offset;
    int64_t partial_lse_offset;
  
    std::vector<int64_t> ToVector() const {
      return {num_blks_x,
              num_blks_y,
              q_indptr_offset,
              kv_indptr_offset,
              partial_indptr_offset,
              merge_packed_offset_start_offset,
              merge_packed_offset_end_offset,
              merge_indptr_offset,
              q_len_offset,
              kv_len_offset,
              q_start_offset,
              kv_start_offset,
              kv_end_offset,
              work_indptr_offset,
              partial_o_offset,
              partial_lse_offset};
    }
  
    void FromVector(const std::vector<int64_t>& vec) {
      if (vec.size() != 16) {
        std::ostringstream err_msg;
        err_msg << "MLAPlanInfo::FromVector: vec.size() should be 12, but got " << vec.size();
        FLASHINFER_ERROR(err_msg.str());
      }
      num_blks_x = vec[0];
      num_blks_y = vec[1];
      q_indptr_offset = vec[2];
      kv_indptr_offset = vec[3];
      partial_indptr_offset = vec[4];
      merge_packed_offset_start_offset = vec[5];
      merge_packed_offset_end_offset = vec[6];
      merge_indptr_offset = vec[7];
      q_len_offset = vec[8];
      kv_len_offset = vec[9];
      q_start_offset = vec[10];
      kv_start_offset = vec[11];
      kv_end_offset = vec[12];
      work_indptr_offset = vec[13];
      partial_o_offset = vec[14];
      partial_lse_offset = vec[15];
    }
  };

template <typename IdType>
inline cudaError_t MLAPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                           void* int_buffer, void* page_locked_int_buffer,
                           size_t int_workspace_size_in_bytes, MLAPlanInfo& plan_info,
                           IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
                           uint32_t batch_size, uint32_t num_heads, uint32_t head_dim_o,
                           bool causal, cudaStream_t stream) {
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // step 0. determine the number of blocks in x and y dimensions
  int accum_packed_qo_len = 0;
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * num_heads;
    accum_packed_qo_len += packed_qo_len;

    int kv_len = kv_len_arr_h[i];
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }
  int avg_packed_qo_len = accum_packed_qo_len / batch_size;

  int cluster_size;
  if (avg_packed_qo_len > 64) {
    cluster_size = 2;  // two ctas in a cluster
  } else {
    cluster_size = 1;  // one cta in a cluster
  }
  uint32_t num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;
  const int cta_tile_q = 64;
  int cluster_tile_q = cluster_size * cta_tile_q;

  int64_t total_kv_lens = 0;
  for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int effective_kv_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx,
                                                           cluster_tile_q, num_qo_tiles, num_heads)
                                    : kv_len;
      total_kv_lens += effective_kv_len;
    }
  }
  int kv_len_limit = ceil_div(std::max(ceil_div(total_kv_lens, num_clusters), 1L), 512L) * 512L;

  // step 1. load-balancing scheduling algorithm
  MinHeap cluster_cost_heap(num_clusters);
  std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
      cluster_kv_indptr(num_clusters, std::vector<IdType>()),
      cluster_q_len(num_clusters, std::vector<IdType>()),
      cluster_kv_len(num_clusters, std::vector<IdType>()),
      cluster_q_start(num_clusters, std::vector<IdType>()),
      cluster_kv_start(num_clusters, std::vector<IdType>()),
      cluster_kv_end(num_clusters, std::vector<IdType>()),
      cluster_partial_indptr(num_clusters, std::vector<IdType>());

  std::vector<IdType> merge_packed_offset_start(num_clusters, 0),
      merge_packed_offset_end(num_clusters, 0), merge_indptr(num_clusters + 1, 0);

  int partial_o_nnz = 0;
  int split_kv_count = 0;

  for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int remaining_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                                        num_qo_tiles, num_heads)
                                 : kv_len;
      int kv_start = 0;
      bool split_kv = remaining_len > kv_len_limit;
      if (split_kv) {
        merge_indptr[split_kv_count] = partial_o_nnz;
        merge_packed_offset_start[split_kv_count] =
            qo_indptr_h[i] * num_heads + qo_tile_idx * cluster_tile_q;
        merge_packed_offset_end[split_kv_count] =
            qo_indptr_h[i] * num_heads +
            std::min((qo_tile_idx + 1) * cluster_tile_q, packed_qo_len);
      }
      bool zero_kv_len = (remaining_len == 0);
      while (remaining_len > 0 || zero_kv_len) {
        auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
        int actual_len = std::min(remaining_len, kv_len_limit);
        cluster_cost_heap.insert(
            {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
        cluster_q_len[cluster_idx].push_back(qo_len);
        cluster_kv_len[cluster_idx].push_back(kv_len);
        cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
        cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);
        if (split_kv) {
          cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);
          partial_o_nnz += std::min(packed_qo_len - qo_tile_idx * cluster_tile_q, cluster_tile_q);
        } else {
          cluster_partial_indptr[cluster_idx].push_back(-1);
        }
        cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
        cluster_kv_start[cluster_idx].push_back(kv_start);
        cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
        remaining_len -= actual_len;
        kv_start += actual_len;
        if (zero_kv_len) break;
      }
      split_kv_count += int(split_kv);
    }
  }
  merge_indptr[split_kv_count] = partial_o_nnz;

  int max_total_num_works = 16384;  // NOTE(Zihao): adjust it later

  std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
  for (uint32_t i = 0; i < num_clusters; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
  }
  int total_num_works = work_indptr_vec.back();
  auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
  auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
  auto q_len_vec = flatten(cluster_q_len, total_num_works);
  auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
  auto q_start_vec = flatten(cluster_q_start, total_num_works);
  auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
  auto kv_end_vec = flatten(cluster_kv_end, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.q_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_indptr");
  plan_info.kv_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_indptr");
  plan_info.partial_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_partial_indptr");
  plan_info.merge_packed_offset_start_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_clusters, 16, "mla_merge_packed_offset_start");
  plan_info.merge_packed_offset_end_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_clusters, 16, "mla_merge_packed_offset_end");
  plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * (num_clusters + 1), 16, "mla_merge_indptr");
  plan_info.q_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_len");
  plan_info.kv_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_len");
  plan_info.q_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_start");
  plan_info.kv_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_start");
  plan_info.kv_end_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_end");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_work_indptr");

  IdType* cluster_q_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_indptr_offset);
  IdType* cluster_kv_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* cluster_partial_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.partial_indptr_offset);
  IdType* cluster_merge_packed_offset_start_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_start_offset);
  IdType* cluster_merge_packed_offset_end_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_end_offset);
  IdType* cluster_merge_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
  IdType* cluster_q_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_len_offset);
  IdType* cluster_kv_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* cluster_q_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_start_offset);
  IdType* cluster_kv_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_start_offset);
  IdType* cluster_kv_end_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_end_offset);
  IdType* cluster_work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);

  std::copy(q_indptr_vec.begin(), q_indptr_vec.end(), cluster_q_indptr_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), cluster_kv_indptr_h);
  std::copy(partial_indptr_vec.begin(), partial_indptr_vec.end(), cluster_partial_indptr_h);
  std::copy(merge_packed_offset_start.begin(), merge_packed_offset_start.end(),
            cluster_merge_packed_offset_start_h);
  std::copy(merge_packed_offset_end.begin(), merge_packed_offset_end.end(),
            cluster_merge_packed_offset_end_h);
  std::copy(merge_indptr.begin(), merge_indptr.end(), cluster_merge_indptr_h);
  std::copy(q_len_vec.begin(), q_len_vec.end(), cluster_q_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), cluster_kv_len_h);
  std::copy(q_start_vec.begin(), q_start_vec.end(), cluster_q_start_h);
  std::copy(kv_start_vec.begin(), kv_start_vec.end(), cluster_kv_start_h);
  std::copy(kv_end_vec.begin(), kv_end_vec.end(), cluster_kv_end_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), cluster_work_indptr_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof(float) * head_dim_o, 16, "mla_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof(float), 16, "mla_partial_lse");

  return cudaSuccess;
}