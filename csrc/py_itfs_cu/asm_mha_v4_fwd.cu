// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "aiter_hip_common.h"
#include "asm_fmha_v4_fwd_configs.hpp"
#include "py_itfs_common.h"
#include "torch/mha_v4_fwd.h"

namespace aiter {
namespace torch_itfs {
namespace {

// These IDs are shared with AttentionFormat in mha_v4.py and the HSA manifests.
enum class AttentionFormat : int64_t
{
    Fp32        = 0,
    Fp16        = 1,
    Bf16        = 2,
    Fp8E4M3     = 3,
    Fp8E4M3Fnuz = 4,
    Fp8E5M2     = 5,
    Fp8E5M2Fnuz = 6,
    Fp6E2M3     = 7,
    Fp6E3M2     = 8,
    Fp4E2M1     = 9,
    Int8        = 10,
    UInt8       = 11,
    Int4        = 12,
    UInt4       = 13,
};

constexpr int64_t format_id(AttentionFormat format) { return static_cast<int64_t>(format); }

// Scale granularity is dispatched independently from the operand encoding.
enum class AttentionScaleMode : int64_t
{
    None          = 0,
    F32PerTensor  = 1,
    F32PerHead    = 2,
    F32PerToken   = 3,
    F32PerChannel = 4,
    E8M0Per1x32   = 5,
};

constexpr int64_t scale_mode_id(AttentionScaleMode mode) { return static_cast<int64_t>(mode); }

constexpr int64_t kHeadDim = 128;

struct PointerSlot
{
    void* value;
    uint32_t padding[2];
};

struct ConstPointerSlot
{
    const void* value;
    uint32_t padding[2];
};

struct ScalarSlot
{
    uint32_t value;
    uint32_t padding[3];
};

// Fixed-width slots reproduce the 656-byte kernarg layout embedded in MHA v4 code objects.
struct __attribute__((packed)) FmhaV4Kernarg
{
    PointerSlot ptr_o;
    ConstPointerSlot ptr_q;
    ConstPointerSlot ptr_k;
    ConstPointerSlot ptr_v;
    PointerSlot ptr_lse;
    ScalarSlot scalar;
    ScalarSlot s_seq_len;
    ScalarSlot s_Seqs;
    ScalarSlot s_Ts;
    ScalarSlot s_Hs;
    ScalarSlot s_Bs;
    ScalarSlot s_gqa;
    ScalarSlot s_k_Seqs;
    ScalarSlot s_k_Hs;
    ScalarSlot s_k_Bs;
    ScalarSlot s_opt;
    ScalarSlot s_lse;
    ScalarSlot s_kv_seq_len;
    ScalarSlot s_qk_head_dim;
    ScalarSlot s_v_head_dim;
    ScalarSlot s_q_head_num;
    ScalarSlot s_v_Seqs;
    ScalarSlot s_v_Hs;
    ScalarSlot s_v_Bs;
    ScalarSlot s_o_Seqs;
    ScalarSlot s_o_Hs;
    ScalarSlot s_o_Bs;
    // Reserved v1 slots keep existing dense code objects at their 656-byte ABI. Sparse, varlen,
    // and LSE support may assign them in later manifest rows; current dense rows leave them zero.
    ConstPointerSlot ptr_qseq;
    ConstPointerSlot ptr_kseq;
    ScalarSlot s_lse_Hs;
    ConstPointerSlot ptr_qseq_padding;
    ConstPointerSlot ptr_kseq_padding;
    ConstPointerSlot ptr_q_descale;
    ConstPointerSlot ptr_k_descale;
    ConstPointerSlot ptr_v_descale;
    ScalarSlot s_descale_q_Bs;
    ScalarSlot s_descale_q_Hs;
    ScalarSlot s_descale_k_Bs;
    ScalarSlot s_descale_k_Hs;
    ScalarSlot s_descale_v_Bs;
    ScalarSlot s_descale_v_Hs;
};

static_assert(sizeof(FmhaV4Kernarg) == 656, "MHA v4 dense kernarg ABI must remain 656 bytes");
static_assert(offsetof(FmhaV4Kernarg, ptr_o) == 0x000);
static_assert(offsetof(FmhaV4Kernarg, ptr_q) == 0x010);
static_assert(offsetof(FmhaV4Kernarg, ptr_k) == 0x020);
static_assert(offsetof(FmhaV4Kernarg, ptr_v) == 0x030);
static_assert(offsetof(FmhaV4Kernarg, scalar) == 0x050);
static_assert(offsetof(FmhaV4Kernarg, ptr_q_descale) == 0x200);
static_assert(offsetof(FmhaV4Kernarg, ptr_k_descale) == 0x210);
static_assert(offsetof(FmhaV4Kernarg, ptr_v_descale) == 0x220);

// Sorted-sparse kernarg: dense 656-byte prefix, LUT pointers at 0x290/0x2A0/0x2B0,
// unused freeze slot at 0x2C0, work_table at 0x2D0, padded to 752.
struct __attribute__((packed)) FmhaV4SparseSortedKernarg
{
    FmhaV4Kernarg dense;
    ConstPointerSlot ptr_kv_block_indices;
    ConstPointerSlot ptr_lut_start;
    ConstPointerSlot ptr_lut_count;
    ConstPointerSlot ptr_lut_freeze;
    ConstPointerSlot ptr_work_table;
    uint32_t s_num_wgs;
    uint32_t s_total_tiles;
    uint64_t tail_pad;
};

static_assert(sizeof(FmhaV4SparseSortedKernarg) == 752,
              "MHA v4 sorted-sparse kernarg ABI must remain 752 bytes");
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_kv_block_indices) == 0x290);
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_lut_start) == 0x2A0);
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_lut_count) == 0x2B0);
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_lut_freeze) == 0x2C0);
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_work_table) == 0x2D0);

void check_format_tensor(const at::Tensor& tensor, int64_t format, const char* name)
{
    if(format == format_id(AttentionFormat::Bf16))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16, name, " must be BF16");
    }
    else if(format == format_id(AttentionFormat::Int8))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Char, name, " must be int8");
    }
    else if(format == format_id(AttentionFormat::Fp8E4M3))
    {
        TORCH_CHECK(
            tensor.scalar_type() == at::ScalarType::Float8_e4m3fn, name, " must be FP8 E4M3 FN");
    }
    else if(format == format_id(AttentionFormat::Fp8E4M3Fnuz))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                    name,
                    " must be FP8 E4M3 FNUZ");
    }
    else if(format == format_id(AttentionFormat::Fp6E2M3) ||
            format == format_id(AttentionFormat::Fp4E2M1))
    {
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Byte,
                    name,
                    " must be a uint8 packed MX tensor");
    }
    else
    {
        TORCH_CHECK(false, "unsupported MHA v4 format id: ", format);
    }
}

const fmha_v4_fwdConfig& find_config(const std::string& arch,
                                     int64_t q_format,
                                     int64_t k_format,
                                     int64_t v_format,
                                     int64_t q_scale_mode,
                                     int64_t k_scale_mode,
                                     int64_t v_scale_mode,
                                     int64_t mode)
{
    for(const auto& entry : cfg_fmha_v4_fwd)
    {
        const auto& cfg = entry.second;
        if(cfg.arch == arch && cfg.q_format == q_format && cfg.k_format == k_format &&
           cfg.v_format == v_format && cfg.q_scale_mode == q_scale_mode &&
           cfg.k_scale_mode == k_scale_mode && cfg.v_scale_mode == v_scale_mode &&
           cfg.o_format == format_id(AttentionFormat::Bf16) &&
           cfg.o_scale_mode == scale_mode_id(AttentionScaleMode::None) && cfg.hdim_q == kHeadDim &&
           cfg.hdim_v == kHeadDim && cfg.mask == 0 && cfg.mode == mode)
            return cfg;
    }
    TORCH_CHECK(false,
                "no MHA v4 kernel for arch=",
                arch,
                ", q_format=",
                q_format,
                ", k_format=",
                k_format,
                ", v_format=",
                v_format,
                ", q_scale_mode=",
                q_scale_mode,
                ", k_scale_mode=",
                k_scale_mode,
                ", v_scale_mode=",
                v_scale_mode,
                ", output=BF16, head_dim=128, mode=",
                mode,
                " (0=dense, 1=sorted-sparse)");
}

void set_descale_strides(const at::Tensor& tensor,
                         int head_dimension,
                         uint32_t& batch_stride,
                         uint32_t& head_stride)
{
    if(tensor.dim() >= 2)
    {
        batch_stride = tensor.stride(0) * tensor.element_size();
        head_stride  = tensor.stride(head_dimension) * tensor.element_size();
    }
}

__device__ inline int32_t pack_work_table_entry(int32_t linear, int32_t q_tiles, int32_t nhead)
{
    const int32_t q_idx = linear % q_tiles;
    const int32_t h_idx = (linear / q_tiles) % nhead;
    const int32_t b_idx = linear / (q_tiles * nhead);
    return (q_idx & 0xFFFF) | ((h_idx & 0xFF) << 16) | ((b_idx & 0xFF) << 24);
}

__global__ void pack_work_table_kernel(int32_t* __restrict__ table,
                                       const int64_t* __restrict__ order,
                                       const int32_t total,
                                       const int32_t q_tiles,
                                       const int32_t nhead)
{
    const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if(slot >= total)
        return;
    table[slot] = pack_work_table_entry(static_cast<int32_t>(order[slot]), q_tiles, nhead);
}

// Largest table the fused path will order. Keys stage in LDS at 8 bytes each, so this is exactly
// the 64KB a workgroup gets and there is no room for a second shared array. Measured build cost
// either side of the limit is in aiter/ops/mha_v4.md.
constexpr int32_t kWorkTableFusedMax    = 8192;
constexpr int32_t kWorkTableSortThreads = 1024;
// Host-side wave width, only used to size the grid. Device code reads warpSize directly, and the
// grid-stride loop stays correct if the two disagree.
constexpr int32_t kWorkTableWave = 64;

// Order and pack in a single launch, by counting each entry's rank rather than moving entries past
// each other. The key carries the whole ordering: the count is complemented into the high half so
// longer LUTs come first, and the slot index sits in the low half, which breaks ties toward raster
// order and so makes uniform counts come out as the identity permutation.
//
// Because the slot index makes every key distinct, an entry's rank is exactly the number of keys
// below it. That is a permutation with no tie-breaking pass, and it is stable by definition.
//
// One wave ranks one entry, each lane counting a strided slice of the keys and the wave reducing
// the partial counts. Total compares are O(n^2), but the work per lane is only n/64 and the only
// barrier is the one after staging, so latency stays near the launch floor over the whole supported
// range. Spreading each entry's count across a wave is what keeps this ahead of ATen's sort here;
// giving one thread a whole entry costs O(n) per thread and loses above about 1024 entries.
//
// A counting sort over the LUT length would be O(n + bins), but positions within a bin would be
// handed out by atomics in arbitrary order. That loses stability, and with it the identity
// permutation for uniform counts, which is the case worth protecting most.
__global__ void rank_and_pack_work_table_kernel(int32_t* __restrict__ table,
                                                const int32_t* __restrict__ lut_count,
                                                const int32_t total,
                                                const int32_t q_tiles,
                                                const int32_t nhead)
{
    __shared__ uint64_t keys[kWorkTableFusedMax];

    for(int32_t slot = threadIdx.x; slot < total; slot += blockDim.x)
    {
        keys[slot] = (static_cast<uint64_t>(~static_cast<uint32_t>(lut_count[slot])) << 32) |
                     static_cast<uint32_t>(slot);
    }
    __syncthreads();

    const int32_t lane            = threadIdx.x % warpSize;
    const int32_t waves_per_block = blockDim.x / warpSize;
    const int32_t first_wave      = blockIdx.x * waves_per_block + threadIdx.x / warpSize;

    // Wave-uniform bounds, so every lane stays active through the reduction below.
    for(int32_t slot = first_wave; slot < total; slot += gridDim.x * waves_per_block)
    {
        const uint64_t mine = keys[slot];
        int32_t rank        = 0;
        for(int32_t other = lane; other < total; other += warpSize)
            rank += keys[other] < mine ? 1 : 0;
        for(int32_t offset = warpSize >> 1; offset > 0; offset >>= 1)
            rank += __shfl_xor(rank, offset);
        if(lane == 0)
            table[rank] = pack_work_table_entry(slot, q_tiles, nhead);
    }
}

at::Tensor
build_sorted_work_table(const at::Tensor& lut_count, int64_t batch, int64_t nhead, int64_t q_tiles)
{
    auto flat           = lut_count.reshape({-1}).contiguous();
    const int64_t total = batch * nhead * q_tiles;
    TORCH_CHECK(flat.numel() == total, "lut_count.numel() must equal batch * heads * query_tiles");
    TORCH_CHECK(batch < 256 && nhead < 256 && q_tiles < 65536,
                "sorted work table packing requires batch<256, heads<256, query_tiles<65536");
    TORCH_CHECK(total <= std::numeric_limits<int32_t>::max(),
                "work table slot index is int32; batch * heads * query_tiles must fit");
    TORCH_CHECK(flat.scalar_type() == at::ScalarType::Int, "lut_count must be int32");

    // LPT order, longest LUT first, so a heavy tile cannot straggle behind the others. Ties resolve
    // toward raster order, which makes the uniform case (top-k sparsity) come out as the identity
    // permutation and keep its spatial locality, with no readback needed to recognise it.
    //
    // This table is rebuilt on every call and its cost does not fall with sparsity, so it otherwise
    // grows to dominate the launch as density drops. Both paths below therefore avoid per-element
    // ATen ops, whose launch overhead on a few hundred entries dwarfs the arithmetic.
    auto table               = at::empty({total}, flat.options().dtype(at::kInt));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(total <= kWorkTableFusedMax)
    {
        // One wave per entry, so the grid follows the table rather than the machine. Every block
        // stages the whole key array, which is why the launch stays one dimension.
        const int32_t waves_per_block = kWorkTableSortThreads / kWorkTableWave;
        const dim3 grid(static_cast<uint32_t>((total + waves_per_block - 1) / waves_per_block));
        rank_and_pack_work_table_kernel<<<grid, dim3(kWorkTableSortThreads), 0, stream>>>(
            table.data_ptr<int32_t>(),
            flat.data_ptr<int32_t>(),
            static_cast<int32_t>(total),
            static_cast<int32_t>(q_tiles),
            static_cast<int32_t>(nhead));
        return table;
    }

    // Past the LDS staging limit, defer the sort to ATen and pack separately. A stable sort keeps
    // the tie behaviour identical to the fused path.
    const auto order = at::argsort(flat, /*stable=*/true, /*dim=*/0, /*descending=*/true);
    constexpr int32_t block_size = 256;
    const dim3 grid(static_cast<uint32_t>((total + block_size - 1) / block_size));
    pack_work_table_kernel<<<grid, dim3(block_size), 0, stream>>>(table.data_ptr<int32_t>(),
                                                                  order.data_ptr<int64_t>(),
                                                                  static_cast<int32_t>(total),
                                                                  static_cast<int32_t>(q_tiles),
                                                                  static_cast<int32_t>(nhead));
    return table;
}

// LUT contents are device data, so the launcher cannot check them without a synchronization. This
// is therefore opt-in: off, an out-of-range index reaches the ASM and faults with only a raw address
// to go on; on, it fails at the launcher with the offending condition named. A row with
// lut_count == 0 is not an error: the ASM skips it and writes zeros for that query tile.
enum LutError : int32_t
{
    kLutNegative   = 1 << 0,
    kLutOverrun    = 1 << 1,
    kLutIndexRange = 1 << 2,
};

__global__ void validate_lut_kernel(const int32_t* __restrict__ lut_start,
                                    const int32_t* __restrict__ lut_count,
                                    const int32_t* __restrict__ kv_block_indices,
                                    const int32_t rows,
                                    const int32_t kv_tiles,
                                    const int64_t kv_index_numel,
                                    int32_t* __restrict__ error)
{
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows)
        return;

    const int32_t start = lut_start[row];
    const int32_t count = lut_count[row];
    if(start < 0 || count < 0)
    {
        atomicOr(error, kLutNegative);
        return;
    }
    // count == 0 is legal and needs no bound: start may sit one past the last entry, and the loop
    // below reads nothing.
    if(static_cast<int64_t>(start) + count > kv_index_numel)
    {
        atomicOr(error, kLutOverrun);
        return;
    }
    for(int32_t i = 0; i < count; ++i)
    {
        const int32_t block = kv_block_indices[start + i];
        if(block < 0 || block >= kv_tiles)
        {
            atomicOr(error, kLutIndexRange);
            return;
        }
    }
}

bool lut_validation_enabled()
{
    static const bool enabled = []() {
        const char* value = std::getenv("AITER_MHA_V4_VALIDATE_LUT");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void validate_lut_contents(const at::Tensor& kv_block_indices,
                           const at::Tensor& lut_start,
                           const at::Tensor& lut_count,
                           int64_t lut_rows,
                           int64_t kv_tiles)
{
    auto error               = at::zeros({1}, lut_count.options().dtype(at::kInt));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    constexpr int32_t block  = 256;
    const dim3 grid(static_cast<uint32_t>((lut_rows + block - 1) / block));
    validate_lut_kernel<<<grid, dim3(block), 0, stream>>>(lut_start.data_ptr<int32_t>(),
                                                          lut_count.data_ptr<int32_t>(),
                                                          kv_block_indices.data_ptr<int32_t>(),
                                                          static_cast<int32_t>(lut_rows),
                                                          static_cast<int32_t>(kv_tiles),
                                                          kv_block_indices.numel(),
                                                          error.data_ptr<int32_t>());
    const int32_t flags = error.cpu().item<int32_t>();
    TORCH_CHECK(!(flags & kLutNegative), "MHA v4 sparse LUT has a negative lut_start or lut_count");
    TORCH_CHECK(!(flags & kLutOverrun),
                "MHA v4 sparse LUT has a row whose lut_start + lut_count exceeds "
                "kv_block_indices.numel() (",
                kv_block_indices.numel(),
                ")");
    TORCH_CHECK(!(flags & kLutIndexRange),
                "MHA v4 sparse LUT has a KV block index outside [0, ",
                kv_tiles,
                ")");
}

void populate_dense_kernarg(FmhaV4Kernarg& args,
                            const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            const at::Tensor& q_descale,
                            const at::Tensor& k_descale,
                            const at::Tensor& v_descale,
                            const at::Tensor& out,
                            const fmha_v4_fwdConfig& cfg,
                            int64_t q_format,
                            int64_t seqlen_q,
                            int64_t seqlen_k,
                            int64_t nhead_q,
                            int64_t gqa_ratio,
                            double softmax_scale)
{
    const bool bf16_format = q_format == format_id(AttentionFormat::Bf16);

    args.ptr_o.value         = out.data_ptr();
    args.ptr_q.value         = q.data_ptr();
    args.ptr_k.value         = k.data_ptr();
    args.ptr_v.value         = v.data_ptr();
    args.ptr_q_descale.value = q_descale.data_ptr();
    args.ptr_k_descale.value = k_descale.data_ptr();
    args.ptr_v_descale.value = v_descale.data_ptr();
    static_assert(sizeof(float) == sizeof(uint32_t));
    const float scale = static_cast<float>(softmax_scale);
    std::memcpy(&args.scalar.value, &scale, sizeof(scale));
    args.s_seq_len.value     = seqlen_q;
    args.s_Seqs.value        = q.stride(1) * q.element_size();
    args.s_Ts.value          = cfg.ts_qo * q.stride(1) * q.element_size();
    args.s_Hs.value          = q.stride(2) * q.element_size();
    args.s_Bs.value          = q.stride(0) * q.element_size();
    args.s_gqa.value         = gqa_ratio;
    args.s_k_Seqs.value      = k.stride(1) * k.element_size();
    args.s_k_Hs.value        = k.stride(2) * k.element_size();
    args.s_k_Bs.value        = k.stride(0) * k.element_size();
    args.s_opt.value         = 5;
    args.s_lse.value         = 0;
    args.s_kv_seq_len.value  = seqlen_k;
    args.s_qk_head_dim.value = kHeadDim;
    args.s_v_head_dim.value  = kHeadDim;
    args.s_q_head_num.value  = nhead_q;
    args.s_v_Seqs.value      = v.stride(1) * v.element_size();
    args.s_v_Hs.value        = v.stride(2) * v.element_size();
    args.s_v_Bs.value        = v.stride(0) * v.element_size();
    args.s_o_Seqs.value      = out.stride(1) * out.element_size();
    args.s_o_Hs.value        = out.stride(2) * out.element_size();
    args.s_o_Bs.value        = out.stride(0) * out.element_size();

    if(!bf16_format)
    {
        set_descale_strides(q_descale,
                            q_descale.dim() >= 3 ? 2 : 1,
                            args.s_descale_q_Bs.value,
                            args.s_descale_q_Hs.value);
        set_descale_strides(k_descale,
                            k_descale.dim() >= 3 ? 2 : 1,
                            args.s_descale_k_Bs.value,
                            args.s_descale_k_Hs.value);
        set_descale_strides(v_descale, 1, args.s_descale_v_Bs.value, args.s_descale_v_Hs.value);
    }
}

struct PackedMhaV4Shapes
{
    int64_t batch;
    int64_t seqlen_q;
    int64_t nhead_q;
    int64_t seqlen_k;
    int64_t nhead_k;
    int64_t gqa_ratio;
};

PackedMhaV4Shapes validate_packed_mha_v4(const at::Tensor& q,
                                         const at::Tensor& k,
                                         const at::Tensor& v,
                                         const at::Tensor& q_descale,
                                         const at::Tensor& k_descale,
                                         const at::Tensor& v_descale,
                                         const at::Tensor& out,
                                         int64_t q_format,
                                         int64_t k_format,
                                         int64_t v_format,
                                         int64_t q_scale_mode,
                                         int64_t k_scale_mode,
                                         int64_t v_scale_mode)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(),
                "Q, K, V, and out must be GPU tensors");
    TORCH_CHECK(q_descale.is_cuda() && k_descale.is_cuda() && v_descale.is_cuda(),
                "all descale tensors must be GPU tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() && q.device() == out.device(),
                "Q, K, V, and out must be on the same GPU");
    TORCH_CHECK(q_descale.device() == q.device() && k_descale.device() == q.device() &&
                    v_descale.device() == q.device(),
                "all descale tensors must be on the same GPU as Q");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
                "MHA v4 expects BSHD tensors");
    TORCH_CHECK(q_format == k_format, "MHA v4 currently requires matching Q/K formats");
    check_format_tensor(q, q_format, "Q");
    check_format_tensor(k, k_format, "K");
    check_format_tensor(v, v_format, "V");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1,
                "Q, K, V, and out must have contiguous last dimensions");

    PackedMhaV4Shapes shapes{};
    shapes.batch               = q.size(0);
    shapes.seqlen_q            = q.size(1);
    shapes.nhead_q             = q.size(2);
    shapes.seqlen_k            = k.size(1);
    shapes.nhead_k             = k.size(2);
    const int64_t packed_width = q_format == format_id(AttentionFormat::Fp6E2M3)   ? 96
                                 : q_format == format_id(AttentionFormat::Fp4E2M1) ? 64
                                                                                   : 128;

    TORCH_CHECK(shapes.batch > 0 && shapes.seqlen_q > 0 && shapes.seqlen_k > 0 &&
                    shapes.nhead_q > 0,
                "MHA v4 requires non-empty inputs");
    TORCH_CHECK(k.size(0) == shapes.batch && v.size(0) == shapes.batch,
                "Q, K, and V batch sizes must match");
    TORCH_CHECK(shapes.nhead_k > 0 && v.size(2) == shapes.nhead_k,
                "MHA v4 requires matching non-empty K and V head dimensions");
    TORCH_CHECK(shapes.nhead_q % shapes.nhead_k == 0,
                "MHA v4 requires query heads to be divisible by KV heads");
    shapes.gqa_ratio = shapes.nhead_q / shapes.nhead_k;
    TORCH_CHECK(shapes.gqa_ratio <= 16 && (shapes.gqa_ratio & (shapes.gqa_ratio - 1)) == 0,
                "MHA v4 supports power-of-two GQA ratios up to 16");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V sequence lengths must match");
    TORCH_CHECK(q.size(3) == packed_width && k.size(3) == packed_width,
                "Q/K packed width does not match the explicit format");
    TORCH_CHECK(v.size(3) == kHeadDim, "V must have logical head dimension 128");
    if(q_format == format_id(AttentionFormat::Fp4E2M1))
    {
        const int64_t tiles       = (shapes.seqlen_k + 127) / 128;
        const int64_t head_stride = tiles * 8192;
        TORCH_CHECK(k.stride(0) == shapes.nhead_k * head_stride && k.stride(1) == 64 &&
                        k.stride(2) == head_stride,
                    "MXFP4 K must use the coalesced MHA v4 tile layout");
    }
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
                "MHA v4 currently supports BF16 output only");
    TORCH_CHECK(out.sizes() ==
                    torch::IntArrayRef({shapes.batch, shapes.seqlen_q, shapes.nhead_q, kHeadDim}),
                "out must have shape [batch, query_length, query_heads, 128]");

    const bool mx_qk_format = q_format == format_id(AttentionFormat::Fp6E2M3) ||
                              q_format == format_id(AttentionFormat::Fp4E2M1);
    const bool bf16_format    = q_format == format_id(AttentionFormat::Bf16);
    const bool e8m0_qk_scales = q_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32) &&
                                k_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32);
    if(bf16_format)
    {
        TORCH_CHECK(q_scale_mode == scale_mode_id(AttentionScaleMode::None) &&
                        k_scale_mode == scale_mode_id(AttentionScaleMode::None) &&
                        v_scale_mode == scale_mode_id(AttentionScaleMode::None),
                    "BF16 Q/K/V must use NONE scale modes");
    }
    else if(e8m0_qk_scales)
    {
        TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Byte &&
                        k_descale.scalar_type() == at::ScalarType::Byte,
                    "MX Q/K descales must be uint8 E8M0 tensors");
        TORCH_CHECK(q_descale.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.seqlen_q, shapes.nhead_q, 4}),
                    "MX Q descale must have shape [batch, query_length, query_heads, 4]");
        TORCH_CHECK(k_descale.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.seqlen_k, shapes.nhead_k, 4}),
                    "MX K descale must have shape [batch, key_length, key_heads, 4]");
    }
    else
    {
        TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Float &&
                        k_descale.scalar_type() == at::ScalarType::Float,
                    "INT8/FP8 Q/K descales must be float32 tensors");
        TORCH_CHECK(q_descale.numel() == 1 && k_descale.numel() == 1,
                    "INT8/FP8 Q/K descales must be scalar tensors");
    }
    const bool mx_v = v_format == format_id(AttentionFormat::Fp6E2M3) ||
                      v_format == format_id(AttentionFormat::Fp4E2M1);
    if(bf16_format)
    {
        // Raw BF16 operands do not use descale tensors.
    }
    else if(mx_v)
    {
        const int64_t tiles = (shapes.seqlen_k + 127) / 128;
        TORCH_CHECK(v_scale_mode == 5 && v_descale.scalar_type() == at::ScalarType::Byte,
                    "MX V descale must use uint8 E8M0 per-1x32 scales");
        TORCH_CHECK(v_descale.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.nhead_k, tiles * 512}),
                    "MX V descale must have shape [batch, key_heads, tiles * 512]");
    }
    else if(mx_qk_format)
    {
        TORCH_CHECK(v_descale.scalar_type() == at::ScalarType::Float,
                    "MX FP8 V descale must be a float32 tensor");
        TORCH_CHECK(v_descale.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.nhead_k, kHeadDim}),
                    "MX V descale must have shape [batch, key_heads, 128]");
    }
    else
    {
        TORCH_CHECK(v_descale.scalar_type() == at::ScalarType::Float,
                    "INT8/FP8 V descale must be a float32 tensor");
        TORCH_CHECK(v_descale.numel() == 1, "INT8/FP8 V descale must be a scalar tensor");
    }
    return shapes;
}

} // namespace

at::Tensor
mha_v4_sparse_work_table(const at::Tensor& lut_count, int64_t batch, int64_t nhead, int64_t q_tiles)
{
    TORCH_CHECK(lut_count.is_cuda(), "lut_count must be a GPU tensor");
    const HipDeviceGuard device_guard{lut_count.get_device()};
    return build_sorted_work_table(lut_count, batch, nhead, q_tiles);
}

void fmha_v4_fwd(const at::Tensor& q,
                 const at::Tensor& k,
                 const at::Tensor& v,
                 const at::Tensor& q_descale,
                 const at::Tensor& k_descale,
                 const at::Tensor& v_descale,
                 at::Tensor out,
                 int64_t q_format,
                 int64_t k_format,
                 int64_t v_format,
                 int64_t q_scale_mode,
                 int64_t k_scale_mode,
                 int64_t v_scale_mode,
                 double softmax_scale)
{
    const auto shapes = validate_packed_mha_v4(q,
                                               k,
                                               v,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               out,
                                               q_format,
                                               k_format,
                                               v_format,
                                               q_scale_mode,
                                               k_scale_mode,
                                               v_scale_mode);

    // Before any device query or launch: get_gpu_arch() reads whichever device is current, and
    // every launch below inherits the current device's stream.
    const HipDeviceGuard device_guard{q.get_device()};

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(
        arch, q_format, k_format, v_format, q_scale_mode, k_scale_mode, v_scale_mode, /*mode=*/0);

    FmhaV4Kernarg args{};
    populate_dense_kernarg(args,
                           q,
                           k,
                           v,
                           q_descale,
                           k_descale,
                           v_descale,
                           out,
                           cfg,
                           q_format,
                           shapes.seqlen_q,
                           shapes.seqlen_k,
                           shapes.nhead_q,
                           shapes.gqa_ratio,
                           softmax_scale);

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel                = kernels.get_or_create(
        cache_key, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    size_t arg_size          = sizeof(args);
    const int gdx            = (shapes.seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int gdy            = shapes.nhead_q;
    const int gdz            = shapes.batch;
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, gdx, gdy, gdz, 512, 1, 1, stream});
}

void fmha_v4_fwd_sparse(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& q_descale,
                        const at::Tensor& k_descale,
                        const at::Tensor& v_descale,
                        at::Tensor out,
                        int64_t q_format,
                        int64_t k_format,
                        int64_t v_format,
                        int64_t q_scale_mode,
                        int64_t k_scale_mode,
                        int64_t v_scale_mode,
                        double softmax_scale,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count)
{
    const auto shapes = validate_packed_mha_v4(q,
                                               k,
                                               v,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               out,
                                               q_format,
                                               k_format,
                                               v_format,
                                               q_scale_mode,
                                               k_scale_mode,
                                               v_scale_mode);

    // Before any device query or launch. build_sorted_work_table() below launches raw HIP kernels,
    // which take the current device and its stream rather than Q's, so an unguarded call on a
    // non-current device faults writing the work table.
    const HipDeviceGuard device_guard{q.get_device()};

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(
        arch, q_format, k_format, v_format, q_scale_mode, k_scale_mode, v_scale_mode, /*mode=*/1);
    TORCH_CHECK(shapes.seqlen_k % cfg.ts_kv == 0,
                "sorted-sparse MHA v4 requires key length padded to a multiple of ",
                cfg.ts_kv);

    const int64_t q_tiles  = (shapes.seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int64_t kv_tiles = (shapes.seqlen_k + cfg.ts_kv - 1) / cfg.ts_kv;
    const int64_t lut_rows = shapes.batch * shapes.nhead_q * q_tiles;
    TORCH_CHECK(kv_block_indices.is_cuda() && lut_start.is_cuda() && lut_count.is_cuda(),
                "LUT tensors must be GPU tensors");
    TORCH_CHECK(kv_block_indices.device() == q.device() && lut_start.device() == q.device() &&
                    lut_count.device() == q.device(),
                "LUT tensors must be on the same GPU as Q");
    TORCH_CHECK(kv_block_indices.scalar_type() == at::ScalarType::Int &&
                    lut_start.scalar_type() == at::ScalarType::Int &&
                    lut_count.scalar_type() == at::ScalarType::Int,
                "LUT tensors must be int32");
    TORCH_CHECK(lut_start.numel() == lut_rows && lut_count.numel() == lut_rows,
                "lut_start and lut_count must have one entry per (batch, head, query tile); "
                "expected ",
                lut_rows,
                " for tile geometry ",
                cfg.ts_qo,
                "x",
                cfg.ts_kv,
                " (",
                q_tiles,
                " x ",
                kv_tiles,
                ")");
    TORCH_CHECK(kv_block_indices.dim() == 1 && lut_start.dim() == 1 && lut_count.dim() == 1,
                "LUT tensors must be 1-D");
    // A row may select nothing, so the entry count is not bounded below by lut_rows. What the ASM
    // still requires is a dereferenceable row base: an empty row is clamped to element 0, and the
    // kernels also read speculatively up to one entry past the row they are traversing (that read
    // is discarded). Per-row bounds need device data and stay behind AITER_MHA_V4_VALIDATE_LUT.
    TORCH_CHECK(kv_block_indices.numel() >= 1,
                "kv_block_indices must be non-empty; the sparse kernels dereference the row base "
                "even for a row that selects no KV block");

    const auto kv_idx = kv_block_indices.contiguous();
    const auto start  = lut_start.contiguous();
    const auto count  = lut_count.contiguous();
    if(lut_validation_enabled())
    {
        validate_lut_contents(kv_idx, start, count, lut_rows, kv_tiles);
    }
    auto work_table = build_sorted_work_table(count, shapes.batch, shapes.nhead_q, q_tiles);

    FmhaV4SparseSortedKernarg args{};
    populate_dense_kernarg(args.dense,
                           q,
                           k,
                           v,
                           q_descale,
                           k_descale,
                           v_descale,
                           out,
                           cfg,
                           q_format,
                           shapes.seqlen_q,
                           shapes.seqlen_k,
                           shapes.nhead_q,
                           shapes.gqa_ratio,
                           softmax_scale);
    args.ptr_kv_block_indices.value = kv_idx.data_ptr();
    args.ptr_lut_start.value        = start.data_ptr();
    args.ptr_lut_count.value        = count.data_ptr();
    args.ptr_work_table.value       = work_table.data_ptr();
    args.s_num_wgs                  = static_cast<uint32_t>(lut_rows);
    args.s_total_tiles              = static_cast<uint32_t>(lut_rows);

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel                = kernels.get_or_create(
        cache_key, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    size_t arg_size          = sizeof(args);
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, static_cast<int>(lut_rows), 1, 1, 512, 1, 1, stream});
}

} // namespace torch_itfs
} // namespace aiter