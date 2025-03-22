#include <fmt/core.h>
#include "pa_ragged.h"
#include "../utils.h"

#define MD_NAME "pa_ragged"

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

void paged_attention_ragged(
    int num_seqs, int num_kv_heads, int num_heads, int max_num_partitions,
    int q_stride, int kv_block_stride, int kv_head_stride, int kv_seq_stride,
    int gqa_ratio, int head_size, std::string dtype, std::string kv_dtype,
    std::string kv_cache_dtype, std::string out_dtype, int block_size,
    std::string alibi_enabled, void *query_ptr, void *key_cache_ptr,
    void *value_cache_ptr, void *workspace_buffer_ptr, int *kv_indptr_ptr,
    int *kv_page_indices_ptr, int *kv_last_page_lens_ptr,
    const float *k_scale_ptr, const float *v_scale_ptr,
    const float *fp8_out_scale_ptr, void *out_ptr,
    const float *alibi_slopes_ptr, float logits_soft_cap, double scale,
    const void *stream) {
  int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warpSize);
  init_root_dir();
  std::string folder = fmt::format(
      "{md_name}_{gqa_ratio}_{head_size}_{npar_loops}_{dtype}_{kv_dtype}_{fp8_"
      "kv_dtype}_{out_dtype}_{block_size}_{alibi_enabled}",
      fmt::arg("md_name", MD_NAME), fmt::arg("gqa_ratio", gqa_ratio),
      fmt::arg("head_size", head_size), fmt::arg("npar_loops", npar_loops),
      fmt::arg("dtype", dtype), fmt::arg("kv_dtype", kv_dtype),
      fmt::arg("fp8_kv_dtype", kv_cache_dtype), fmt::arg("out_dtype", dtype),
      fmt::arg("block_size", block_size),
      fmt::arg("alibi_enabled", alibi_enabled));
  if (!std::filesystem::exists(get_root_dir() / "build" / folder / "lib.so")) {
    std::string cmd = fmt::format(
        R"(python3 pa_ragged.py --gqa_ratio={gqa_ratio} \
                                    --head_size={head_size} \
                                    --npar_loops={npar_loops} \
                                    --dtype={dtype} \
                                    --kv_dtype={kv_dtype} \
                                    --fp8_kv_dtype={fp8_kv_dtype} \
                                    --out_dtype={out_dtype} \
                                    --block_size={block_size} \
                                    --alibi_enabled={alibi_enabled})",
        fmt::arg("gqa_ratio", gqa_ratio), fmt::arg("head_size", head_size),
        fmt::arg("npar_loops", npar_loops), fmt::arg("dtype", dtype),
        fmt::arg("kv_dtype", kv_dtype),
        fmt::arg("fp8_kv_dtype", kv_cache_dtype), fmt::arg("out_dtype", dtype),
        fmt::arg("block_size", block_size),
        fmt::arg("alibi_enabled", alibi_enabled));
    executeCmd(cmd);
  }

  run_lib(folder, out_ptr, workspace_buffer_ptr, query_ptr, key_cache_ptr,
          value_cache_ptr, scale, num_seqs, num_kv_heads, num_heads,
          max_num_partitions, q_stride, kv_block_stride, kv_head_stride,
          kv_seq_stride, kv_indptr_ptr, kv_page_indices_ptr,
          kv_last_page_lens_ptr, alibi_slopes_ptr, logits_soft_cap, k_scale_ptr,
          v_scale_ptr, fp8_out_scale_ptr, stream);
}