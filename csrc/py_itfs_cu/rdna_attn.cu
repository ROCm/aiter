// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The RDNA attention ops of rdna_attn.h. rdna_*_configs list a table's
// entries and rdna_{decode,prefill,encoder}_attn launch a build by its index,
// both in the tables of a device's GPU family (rdna_attn_types.h). The
// launch functions are registered by the module's units
// (csrc/kernels/rdna_attn/rdna_attn_unit.cuh), for one head dim and dtype.
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aiter_stream.h"
#include "rdna_attn.h"
#include "rdna_attn_launch.h"
#include "rdna_decode_batch_configs.h"
#include "rdna_decode_configs.h"
#include "rdna_prefill_batch_configs.h"
#include "rdna_prefill_configs.h"

rdna::LaunchFn* rdna::decode_launchers(Arch arch) {
  static LaunchFn gfx1151[std::size(kGfx1151DecodeConfigs)] = {};
  static LaunchFn gfx1150[std::size(kGfx1150DecodeConfigs)] = {};
  static LaunchFn gfx1201[std::size(kGfx1201DecodeConfigs)] = {};
  return arch == Arch::kGfx1151   ? gfx1151
         : arch == Arch::kGfx1150 ? gfx1150
                                  : gfx1201;
}

rdna::LaunchFn* rdna::decode_batch_launchers(Arch arch) {
  static LaunchFn gfx1151[std::size(kGfx1151DecodeBatchConfigs)] = {};
  static LaunchFn gfx1150[std::size(kGfx1150DecodeBatchConfigs)] = {};
  static LaunchFn gfx1201[std::size(kGfx1201DecodeBatchConfigs)] = {};
  return arch == Arch::kGfx1151   ? gfx1151
         : arch == Arch::kGfx1150 ? gfx1150
                                  : gfx1201;
}

rdna::PrefillFn* rdna::prefill_launchers(Arch arch) {
  static PrefillFn gfx1151[std::size(kGfx1151PrefillConfigs)] = {};
  static PrefillFn gfx1150[std::size(kGfx1150PrefillConfigs)] = {};
  static PrefillFn gfx1201[std::size(kGfx1201PrefillConfigs)] = {};
  return arch == Arch::kGfx1151   ? gfx1151
         : arch == Arch::kGfx1150 ? gfx1150
                                  : gfx1201;
}

namespace {

using rdna::Arch;
using rdna::DecodeConfig;
using rdna::LaunchFn;
using rdna::PrefillBatchConfig;
using rdna::PrefillConfig;
using rdna::PrefillFn;

// A GPU family's tables and their launch functions.
struct Table {
  const DecodeConfig* decode = nullptr;
  int64_t num_decode = 0;
  LaunchFn* decode_launchers = nullptr;
  const DecodeConfig* decode_batch = nullptr;
  int64_t num_decode_batch = 0;
  LaunchFn* decode_batch_launchers = nullptr;
  const PrefillConfig* prefill = nullptr;
  int64_t num_prefill = 0;
  PrefillFn* prefill_launchers = nullptr;
  const PrefillBatchConfig* prefill_batch = nullptr;
  int64_t num_prefill_batch = 0;
};

Table table_of(Arch arch) {
  const int a = static_cast<int>(arch);
  const auto decode = rdna::kDecodeConfigs[a];
  const auto decode_batch = rdna::kDecodeBatchConfigs[a];
  const auto prefill = rdna::kPrefillConfigs[a];
  const auto prefill_batch = rdna::kPrefillBatchConfigs[a];
  return {decode.data(),
          static_cast<int64_t>(decode.size()),
          rdna::decode_launchers(arch),
          decode_batch.data(),
          static_cast<int64_t>(decode_batch.size()),
          rdna::decode_batch_launchers(arch),
          prefill.data(),
          static_cast<int64_t>(prefill.size()),
          rdna::prefill_launchers(arch),
          prefill_batch.data(),
          static_cast<int64_t>(prefill_batch.size())};
}

// Raises a std::runtime_error (a Python RuntimeError) of the streamed
// arguments unless `cond`.
template <typename... Args>
[[noreturn]] void fail(const Args&... args) {
  std::ostringstream message;
  (message << ... << args);
  throw std::runtime_error(message.str());
}

#define RDNA_CHECK(cond, ...)                 \
  do {                                        \
    if (!(cond)) [[unlikely]]                 \
      fail("rdna_attn: ", __VA_ARGS__);       \
  } while (0)

std::string sizes(const aiter_tensor_t& t) {
  std::ostringstream text;
  text << "[";
  for (int i = 0; i < t.dim(); ++i) text << (i ? ", " : "") << t.size(i);
  text << "]";
  return text.str();
}

std::string strides(const aiter_tensor_t& t) {
  std::ostringstream text;
  text << "[";
  for (int i = 0; i < t.dim(); ++i) text << (i ? ", " : "") << t.stride(i);
  text << "]";
  return text.str();
}

// The properties of a device, read once.
const hipDeviceProp_t& device_properties(int64_t device_id) {
  static constexpr int kMaxDevices = 64;
  static hipDeviceProp_t props[kMaxDevices];
  static bool read[kMaxDevices] = {};
  RDNA_CHECK(device_id >= 0 && device_id < kMaxDevices, "no GPU ", device_id);
  if (!read[device_id]) {
    RDNA_CHECK(hipGetDeviceProperties(&props[device_id],
                                      static_cast<int>(device_id)) == hipSuccess,
               "cannot read the properties of GPU ", device_id);
    read[device_id] = true;
  }
  return props[device_id];
}

// The tables of a device's GPU family; empty on other GPUs.
Table device_table(int64_t device_id) {
  const char* name = device_properties(device_id).gcnArchName;
  const auto is = [&](const char* arch) {
    return std::strstr(name, arch) != nullptr;
  };
  if (is("gfx1151")) return table_of(Arch::kGfx1151);
  if (is("gfx1150") || is("gfx1152") || is("gfx1153"))
    return table_of(Arch::kGfx1150);
  if (is("gfx1200") || is("gfx1201")) return table_of(Arch::kGfx1201);
  return {};
}

// The tables of the device a launch's tensors are on.
Table device_table(const aiter_tensor_t& q) {
  RDNA_CHECK(q.is_gpu(), "q must be on a GPU");
  return device_table(q.device_id);
}

// Why a table's entry cannot launch here.
const char* kNoUnit =
    ": this module is not built for its head dim, dtype and GPU family "
    "(aiter/ops/rdna_attn.py picks the module)";

// The kernels expect (num_pages, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM).
void check_kv_cache(const aiter_tensor_t& kv_cache, int num_kv_heads,
                    int page_size, int head_dim) {
  const int64_t row = 2 * head_dim;
  RDNA_CHECK(kv_cache.dim() == 4 && kv_cache.size(1) == num_kv_heads &&
                 kv_cache.size(2) == page_size && kv_cache.size(3) == row &&
                 kv_cache.stride(3) == 1 && kv_cache.stride(2) == row &&
                 kv_cache.stride(1) == page_size * row &&
                 kv_cache.stride(0) == num_kv_heads * page_size * row,
             "kv_cache must be packed HND (pages, ", num_kv_heads, ", ",
             page_size, ", ", row, "), got sizes ", sizes(kv_cache),
             " strides ", strides(kv_cache));
}

void check_device(const aiter_tensor_t& q,
                  std::initializer_list<const aiter_tensor_t*> others) {
  for (const aiter_tensor_t* t : others)
    RDNA_CHECK(t->device_id == q.device_id, "every tensor must be on GPU ",
               q.device_id, ", got one on device ", t->device_id);
}

bool same_sizes(const aiter_tensor_t& a, const aiter_tensor_t& b) {
  if (a.dim() != b.dim()) return false;
  for (int i = 0; i < a.dim(); ++i)
    if (a.size(i) != b.size(i)) return false;
  return true;
}

const int* ints(const aiter_tensor_t& t) {
  return static_cast<const int*>(t.data_ptr());
}

int* ints(aiter_tensor_t& t) { return static_cast<int*>(t.data_ptr()); }

float* floats(aiter_tensor_t& t) { return static_cast<float*>(t.data_ptr()); }

// Every entry of a decode table, its DecodeConfig fields in order.
std::vector<int64_t> decode_configs(const DecodeConfig* configs, int64_t n) {
  std::vector<int64_t> fields;
  for (int64_t i = 0; i < n; ++i) {
    const DecodeConfig& c = configs[i];
    fields.insert(
        fields.end(),
        {static_cast<int>(c.head_dim), c.num_q_heads, c.num_kv_heads, c.window,
         c.max_query_len, static_cast<int>(c.dtype), c.page_size,
         c.max_segments, c.row_groups, c.min_segment_blocks, c.waves,
         c.head_dim_split, c.prefetch, c.v_in_lds, c.dot_product});
  }
  return fields;
}

// The split-KV scratch a prefill launch may write, as
// PrefillVariant.scratch_rows.
void check_prefill_scratch(const PrefillConfig& c, const aiter_tensor_t& q,
                           const aiter_tensor_t& partial_o,
                           const aiter_tensor_t& partial_ml,
                           const aiter_tensor_t& counters) {
  if (c.max_segments <= 1) return;
  const int64_t d = static_cast<int>(c.head_dim);
  const int64_t target =
      4 * device_properties(q.device_id).multiProcessorCount;
  const int64_t block_rows =
      c.waves / (c.head_dim_split * c.key_waves) * 16 * c.row_tiles;
  const int64_t rows = 2 * target * block_rows;
  RDNA_CHECK((int64_t)partial_o.numel() >= rows * d &&
                 (int64_t)partial_ml.numel() >= 2 * rows &&
                 (int64_t)counters.numel() >= target,
             "scratch too small for this build: needs ", rows * d, ", ",
             2 * rows, ", ", target, " elements");
}

}  // namespace

std::vector<int64_t> rdna_decode_configs(int64_t device_id) {
  const Table t = device_table(device_id);
  return decode_configs(t.decode, t.num_decode);
}

std::vector<int64_t> rdna_decode_batch_configs(int64_t device_id) {
  const Table t = device_table(device_id);
  return decode_configs(t.decode_batch, t.num_decode_batch);
}

std::vector<int64_t> rdna_prefill_configs(int64_t device_id) {
  std::vector<int64_t> fields;
  const Table t = device_table(device_id);
  for (int64_t i = 0; i < t.num_prefill; ++i) {
    const PrefillConfig& c = t.prefill[i];
    fields.insert(
        fields.end(),
        {static_cast<int>(c.head_dim), c.num_q_heads, c.num_kv_heads, c.window,
         static_cast<int>(c.dtype), c.page_size, c.min_query_len, c.waves,
         c.key_tile, c.head_dim_split, c.v_t, c.double_buffer, c.key_waves,
         c.max_segments, c.head_group, c.causal, c.row_tiles});
  }
  return fields;
}

// Raises if a cell names no prefill build.
std::vector<int64_t> rdna_prefill_batch_configs(int64_t device_id) {
  using namespace rdna;
  std::vector<int64_t> fields;
  const Table t = device_table(device_id);
  for (int64_t i = 0; i < t.num_prefill_batch; ++i) {
    const PrefillBatchConfig& b = t.prefill_batch[i];
    for (Dtype dtype : {Dtype::kFp16, Dtype::kBf16}) {
      const bool built = std::any_of(
          t.prefill, t.prefill + t.num_prefill, [&](const PrefillConfig& c) {
            return c.head_dim == b.head_dim && c.num_q_heads == b.num_q_heads &&
                   c.num_kv_heads == b.num_kv_heads && c.window == b.window &&
                   c.page_size == b.page_size && c.dtype == dtype &&
                   c.min_query_len == b.min_query_len && c.causal == b.causal;
          });
      RDNA_CHECK(built, "prefill batch cell of ", b.num_q_heads, "/",
                 b.num_kv_heads, "/", static_cast<int>(b.head_dim), " window ",
                 b.window, " page ", b.page_size,
                 " names no prefill build with min_query_len ", b.min_query_len,
                 ", bf16 ", dtype == Dtype::kBf16);
    }
    fields.insert(fields.end(),
                  {static_cast<int>(b.head_dim), b.num_q_heads, b.num_kv_heads,
                   b.window, b.page_size, b.min_mean_m, b.min_keys,
                   b.min_query_len, b.head_group, b.causal});
  }
  return fields;
}

// The caller allocates the scratch: this runs under CUDA-graph capture.
void rdna_decode_attn(aiter_tensor_t& q,
                      aiter_tensor_t& kv_cache, aiter_tensor_t& block_table,
                      aiter_tensor_t& out, aiter_tensor_t& acc,
                      aiter_tensor_t& m, aiter_tensor_t& l, aiter_tensor_t& cnt,
                      aiter_tensor_t& seq_lens, int64_t config, float scale,
                      std::optional<aiter_tensor_t> mm_ranges, bool batched) {
  using namespace rdna;
  const Table t = device_table(q);
  LaunchFn* launchers = batched ? t.decode_batch_launchers : t.decode_launchers;
  RDNA_CHECK(
      config >= 0 && config < (batched ? t.num_decode_batch : t.num_decode),
      "no ", batched ? "batched " : "", "decode build ", config);
  RDNA_CHECK(launchers[config], batched ? "batched " : "", "decode build ",
             config, kNoUnit);
  const DecodeConfig& c = batched ? t.decode_batch[config] : t.decode[config];
  const int maxm = c.max_query_len;
  RDNA_CHECK(q.is_contiguous() && out.is_contiguous(),
             "q and out must be contiguous");
  RDNA_CHECK(block_table.dtype() == AITER_DTYPE_i32 &&
                 (block_table.dim() == 1 || block_table.dim() == 2) &&
                 block_table.stride(-1) == 1,
             "block table must be int32 rows, contiguous within a row");
  RDNA_CHECK(seq_lens.dtype() == AITER_DTYPE_i32 && seq_lens.numel() >= 1,
             "seq_lens must be an int32 tensor holding the sequence length");
  // nseq sequences; sequence s reads row s of the block table and scratch.
  RDNA_CHECK(q.dim() == 3 && q.size(0) % maxm == 0 && q.size(0) > 0, "q has ",
             q.dim() == 3 ? q.size(0) : -1,
             " tokens, not a multiple of max_query_len=", maxm);
  const int nseq = q.size(0) / maxm;
  RDNA_CHECK(batched || nseq == 1, "one-sequence build launched for ", nseq,
             " sequences");
  RDNA_CHECK((int64_t)seq_lens.numel() >= nseq, "seq_lens has ",
             seq_lens.numel(), " entries for ", nseq, " sequences");
  RDNA_CHECK(block_table.dim() == 2 ? block_table.size(0) >= nseq : nseq == 1,
             "block table has too few rows for ", nseq, " sequences");
  RDNA_CHECK(
      !batched || (acc.dim() == 3 && acc.size(0) >= nseq && m.dim() >= 1 &&
                   m.size(0) >= nseq && l.dim() >= 1 && l.size(0) >= nseq &&
                   cnt.dim() == 2 && cnt.size(0) >= nseq),
      "scratch holds fewer sequences than the ", nseq, " launched");
  RDNA_CHECK(
      q.size(1) == c.num_q_heads && q.size(2) == static_cast<int>(c.head_dim),
      "q shape does not match the build");
  const AiterDtype dtype =
      c.dtype == Dtype::kBf16 ? AITER_DTYPE_bf16 : AITER_DTYPE_fp16;
  RDNA_CHECK(q.dtype() == dtype && kv_cache.dtype() == dtype &&
                 out.dtype() == dtype,
             "kernel built for ", AiterDtype_to_str(dtype),
             ", got q=", AiterDtype_to_str(q.dtype()),
             " kv_cache=", AiterDtype_to_str(kv_cache.dtype()),
             " out=", AiterDtype_to_str(out.dtype()));
  RDNA_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
             " elements, got ", out.numel());
  const int hq = c.num_q_heads, hkv = c.num_kv_heads,
            d = static_cast<int>(c.head_dim);
  check_kv_cache(kv_cache, hkv, c.page_size, d);
  check_device(q,
               {&kv_cache, &block_table, &out, &acc, &m, &l, &cnt, &seq_lens});
  // As KernelVariant.scratch_shapes.
  const int segments = c.max_segments;
  int64_t rows, counters;
  if (c.dot_product) {
    rows = (int64_t)hq * segments * maxm;
    counters = hq;
  } else {
    const int groups = c.row_groups;
    const int64_t padded = ((hq / hkv) * maxm / groups + 15) / 16 * 16;
    rows = (int64_t)hkv * groups * segments * padded;
    counters = 4 * hkv * groups;
  }
  RDNA_CHECK(
      acc.dtype() == AITER_DTYPE_fp32 && m.dtype() == AITER_DTYPE_fp32 &&
          l.dtype() == AITER_DTYPE_fp32 && cnt.dtype() == AITER_DTYPE_i32,
      "scratch must be float32 partials and int32 counters");
  const int64_t seqs = batched ? acc.size(0) : 1;
  RDNA_CHECK((int64_t)acc.numel() / seqs >= rows * d &&
                 (int64_t)m.numel() / seqs >= rows &&
                 (int64_t)l.numel() / seqs >= rows &&
                 (int64_t)cnt.numel() / seqs >= counters,
             "scratch too small for this build: needs ", rows * d, ", ", rows,
             ", ", rows, ", ", counters, " elements");
  // Per sequence, [count, start0, end0, ...] in device memory: CUDA graphs
  // read the counts at replay. One row, or a row per sequence.
  const int* ranges = nullptr;
  int capacity = 0, mm_stride = 0;
  if (mm_ranges.has_value()) {
    const bool per_sequence = mm_ranges->dim() == 2;
    RDNA_CHECK(
        mm_ranges->dtype() == AITER_DTYPE_i32 && mm_ranges->is_contiguous() &&
            (mm_ranges->dim() == 1 || per_sequence) &&
            mm_ranges->size(-1) % 2 == 1 &&
            (per_sequence ? mm_ranges->size(0) >= nseq : nseq == 1) &&
            mm_ranges->device_id == q.device_id,
        "mm_ranges must be a contiguous int32 [count, start0, end0, "
        "...], or one such row per sequence, on GPU ",
        q.device_id, ", got ", AiterDtype_to_str(mm_ranges->dtype()), " ",
        sizes(*mm_ranges), " on device ", mm_ranges->device_id, " for ", nseq,
        " sequences");
    ranges = ints(*mm_ranges);
    capacity = static_cast<int>(mm_ranges->size(-1) / 2);
    mm_stride =
        batched && per_sequence ? static_cast<int>(mm_ranges->stride(0)) : 0;
  }
  const HipDeviceGuard device_guard(q.device_id);
  launchers[config]({q.data_ptr(),
                     kv_cache.data_ptr(),
                     ints(block_table),
                     floats(acc),
                     floats(m),
                     floats(l),
                     ints(cnt),
                     out.data_ptr(),
                     ints(seq_lens),
                     (int)block_table.size(-1),
                     scale,
                     aiter::getCurrentHIPStream(),
                     ranges,
                     capacity,
                     nseq,
                     block_table.dim() == 2 ? (int)block_table.stride(0) : 0,
                     batched ? (int)acc.stride(0) : 0,
                     batched ? (int)m.stride(0) : 0,
                     batched ? (int)cnt.stride(0) : 0,
                     mm_stride});
}

// One sequence, or a batch with a plan; runs outside CUDA graphs, so
// max_seq_len may size the split.
void rdna_prefill_attn(aiter_tensor_t& q,
                       aiter_tensor_t& kv_cache, aiter_tensor_t& block_table,
                       aiter_tensor_t& out, aiter_tensor_t& partial_o,
                       aiter_tensor_t& partial_ml, aiter_tensor_t& counters,
                       aiter_tensor_t& seq_lens, int64_t config, int64_t max_seq_len,
                       float scale, std::optional<aiter_tensor_t> mm_ranges,
                       std::optional<aiter_tensor_t> plan,
                       std::optional<aiter_tensor_t> query_start_loc,
                       int64_t head_group) {
  using namespace rdna;
  const Table t = device_table(q);
  RDNA_CHECK(config >= 0 && config < t.num_prefill, "no prefill build ",
             config);
  RDNA_CHECK(t.prefill_launchers[config], "prefill build ", config, kNoUnit);
  const PrefillConfig& c = t.prefill[config];
  RDNA_CHECK(c.causal, "a non-causal build runs through rdna_encoder_attn");
  RDNA_CHECK(q.dim() == 3 && q.is_contiguous() && out.is_contiguous(),
             "q and out must be contiguous (tokens, heads, head dim)");
  RDNA_CHECK(block_table.dtype() == AITER_DTYPE_i32 &&
                 (block_table.dim() == 1 || block_table.dim() == 2) &&
                 block_table.stride(-1) == 1,
             "block table must be an int32 row, contiguous");
  RDNA_CHECK(seq_lens.dtype() == AITER_DTYPE_i32 && seq_lens.numel() >= 1,
             "seq_lens must be an int32 tensor holding the sequence length");
  RDNA_CHECK(plan.has_value() == query_start_loc.has_value(),
             "a batch takes both a plan and query_start_loc");
  if (plan) {
    RDNA_CHECK(plan->dtype() == AITER_DTYPE_i32 && plan->dim() == 2 &&
                   plan->size(1) == 4 && plan->is_contiguous(),
               "plan must be plan_prefill's contiguous int32 [entries, 4]");
    RDNA_CHECK(query_start_loc->dtype() == AITER_DTYPE_i32 &&
                   query_start_loc->numel() >= 2,
               "query_start_loc must be int32, one entry per sequence and one");
    const int64_t nseq = query_start_loc->numel() - 1;
    RDNA_CHECK(block_table.dim() == 2 && block_table.size(0) >= nseq &&
                   (int64_t)seq_lens.numel() >= nseq,
               "block_table and seq_lens need a row per sequence");
    RDNA_CHECK(head_group >= 0 &&
                   (head_group == 0 || c.num_kv_heads % head_group == 0),
               "head_group must divide the kv heads");
  } else {
    RDNA_CHECK(q.size(0) > 0 && max_seq_len >= q.size(0),
               "max_seq_len must cover the query tokens");
  }
  RDNA_CHECK(
      q.size(1) == c.num_q_heads && q.size(2) == static_cast<int>(c.head_dim),
      "q shape does not match the build");
  RDNA_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 &&
                 partial_ml.dtype() == AITER_DTYPE_fp32 &&
                 counters.dtype() == AITER_DTYPE_i32,
             "scratch must be float32 partials and int32 counters");
  const AiterDtype dtype =
      c.dtype == Dtype::kBf16 ? AITER_DTYPE_bf16 : AITER_DTYPE_fp16;
  RDNA_CHECK(q.dtype() == dtype && kv_cache.dtype() == dtype &&
                 out.dtype() == dtype,
             "kernel built for ", AiterDtype_to_str(dtype),
             ", got q=", AiterDtype_to_str(q.dtype()),
             " kv_cache=", AiterDtype_to_str(kv_cache.dtype()),
             " out=", AiterDtype_to_str(out.dtype()));
  RDNA_CHECK(out.numel() == q.numel(), "out must hold q's ", q.numel(),
             " elements, got ", out.numel());
  const int d = static_cast<int>(c.head_dim);
  check_kv_cache(kv_cache, c.num_kv_heads, c.page_size, d);
  check_device(q, {&kv_cache, &block_table, &out, &partial_o, &partial_ml,
                   &counters, &seq_lens});
  if (plan) check_device(q, {&*plan, &*query_start_loc});
  const int* ranges = nullptr;
  int num_ranges = 0;
  if (mm_ranges.has_value()) {
    // One sequence's (ranges, 2), or with a plan the batch's
    // (sequences, ranges, 2).
    const int dim = plan ? 3 : 2;
    RDNA_CHECK(
        mm_ranges->dtype() == AITER_DTYPE_i32 && mm_ranges->is_contiguous() &&
            mm_ranges->dim() == dim && mm_ranges->size(-1) == 2 &&
            mm_ranges->device_id == q.device_id &&
            (!plan ||
             mm_ranges->size(0) >= (int64_t)query_start_loc->numel() - 1),
        "mm_ranges must be contiguous int32 ", plan ? "(sequences, " : "(",
        "ranges, 2) on GPU ", q.device_id, ", got ",
        AiterDtype_to_str(mm_ranges->dtype()), " ", sizes(*mm_ranges),
        " on device ", mm_ranges->device_id);
    ranges = ints(*mm_ranges);
    num_ranges = static_cast<int>(mm_ranges->size(-2));
  }
  check_prefill_scratch(c, q, partial_o, partial_ml, counters);
  const HipDeviceGuard device_guard(q.device_id);
  PrefillArgs args{};
  args.q = q.data_ptr();
  args.kv = kv_cache.data_ptr();
  args.bt = ints(block_table);
  args.out = out.data_ptr();
  args.seq_lens = ints(seq_lens);
  args.partial_o = floats(partial_o);
  args.partial_ml = floats(partial_ml);
  args.counters = ints(counters);
  args.num_tokens = (int)q.size(0);
  args.max_seq_len = (int)max_seq_len;
  args.scale = scale;
  args.stream = aiter::getCurrentHIPStream();
  args.mm_ranges = ranges;
  args.num_mm_ranges = num_ranges;
  args.plan = plan ? ints(*plan) : nullptr;
  args.query_start = plan ? ints(*query_start_loc) : nullptr;
  args.num_entries = plan ? (int)plan->size(0) : 0;
  args.bt_stride = block_table.dim() == 2 ? (int)block_table.stride(0) : 0;
  args.head_group = (int)head_group;
  t.prefill_launchers[config](args);
}

void rdna_encoder_attn(aiter_tensor_t& q, aiter_tensor_t& k,
                       aiter_tensor_t& v, aiter_tensor_t& out,
                       aiter_tensor_t& partial_o, aiter_tensor_t& partial_ml,
                       aiter_tensor_t& counters, aiter_tensor_t& seq_lens,
                       int64_t config,
                       int64_t max_seq_len, float scale,
                       std::optional<aiter_tensor_t> plan,
                       std::optional<aiter_tensor_t> query_start_loc,
                       int64_t head_group) {
  using namespace rdna;
  const Table t = device_table(q);
  RDNA_CHECK(config >= 0 && config < t.num_prefill, "no prefill build ",
             config);
  RDNA_CHECK(t.prefill_launchers[config], "prefill build ", config, kNoUnit);
  const PrefillConfig& c = t.prefill[config];
  RDNA_CHECK(!c.causal, "a causal build runs through rdna_prefill_attn");
  const int d = static_cast<int>(c.head_dim);
  RDNA_CHECK(q.dim() == 3, "q must be (tokens, heads, head dim)");
  // (tokens, heads, D), each token's heads contiguous; any token stride.
  auto check = [&](const aiter_tensor_t& x, int heads, const char* name) {
    RDNA_CHECK(x.dim() == 3 && x.size(0) == q.size(0) && x.size(1) == heads &&
                   x.size(2) == d && x.stride(2) == 1 && x.stride(1) == d,
               name, " must be (", q.size(0), ", ", heads, ", ", d,
               ") with each token's heads contiguous, got ", sizes(x),
               " strides ", strides(x));
  };
  check(q, c.num_q_heads, "q");
  check(k, c.num_kv_heads, "k");
  check(v, c.num_kv_heads, "v");
  RDNA_CHECK(out.is_contiguous() && same_sizes(out, q),
             "out must be contiguous and shaped as q");
  const AiterDtype dtype =
      c.dtype == Dtype::kBf16 ? AITER_DTYPE_bf16 : AITER_DTYPE_fp16;
  RDNA_CHECK(q.dtype() == dtype && k.dtype() == dtype && v.dtype() == dtype &&
                 out.dtype() == dtype,
             "kernel built for ", AiterDtype_to_str(dtype));
  RDNA_CHECK(seq_lens.dtype() == AITER_DTYPE_i32 && seq_lens.numel() >= 1,
             "seq_lens must be int32, a length per sequence");
  RDNA_CHECK(plan.has_value() == query_start_loc.has_value(),
             "a batch takes both a plan and query_start_loc");
  if (plan) {
    RDNA_CHECK(plan->dtype() == AITER_DTYPE_i32 && plan->dim() == 2 &&
                   plan->size(1) == 4 && plan->is_contiguous(),
               "plan must be plan_prefill's contiguous int32 [entries, 4]");
    RDNA_CHECK(query_start_loc->dtype() == AITER_DTYPE_i32 &&
                   query_start_loc->numel() >= 2 &&
                   seq_lens.numel() >= query_start_loc->numel() - 1,
               "query_start_loc must be int32, one entry per sequence and "
               "one, with a length per sequence");
    RDNA_CHECK(head_group >= 0 &&
                   (head_group == 0 || c.num_kv_heads % head_group == 0),
               "head_group must divide the kv heads");
  } else {
    RDNA_CHECK(q.size(0) > 0 && max_seq_len == q.size(0),
               "one sequence: max_seq_len is its tokens");
  }
  RDNA_CHECK(partial_o.dtype() == AITER_DTYPE_fp32 &&
                 partial_ml.dtype() == AITER_DTYPE_fp32 &&
                 counters.dtype() == AITER_DTYPE_i32,
             "scratch must be float32 partials and int32 counters");
  check_device(q,
               {&k, &v, &out, &partial_o, &partial_ml, &counters, &seq_lens});
  if (plan) check_device(q, {&*plan, &*query_start_loc});
  check_prefill_scratch(c, q, partial_o, partial_ml, counters);
  const HipDeviceGuard device_guard(q.device_id);
  PrefillArgs args{};
  args.q = q.data_ptr();
  args.kv = k.data_ptr();
  args.out = out.data_ptr();
  args.seq_lens = ints(seq_lens);
  args.partial_o = floats(partial_o);
  args.partial_ml = floats(partial_ml);
  args.counters = ints(counters);
  args.num_tokens = (int)q.size(0);
  args.max_seq_len = (int)max_seq_len;
  args.scale = scale;
  args.stream = aiter::getCurrentHIPStream();
  args.plan = plan ? ints(*plan) : nullptr;
  args.query_start = plan ? ints(*query_start_loc) : nullptr;
  args.num_entries = plan ? (int)plan->size(0) : 0;
  args.head_group = (int)head_group;
  args.v = v.data_ptr();
  args.q_stride = (int)q.stride(0);
  args.k_stride = (int)k.stride(0);
  args.v_stride = (int)v.stride(0);
  t.prefill_launchers[config](args);
}
