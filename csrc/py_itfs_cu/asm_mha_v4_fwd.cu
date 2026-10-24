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

// V's layout within its format, shared with AttentionPack in mha_v4.py and the manifest's v_pack
// column. Two rows can take the same format and scale mode in different layouts (MXFP4 V in the
// column-major order or in the order an FP6 P operand contracts over), so it is part of the key.
enum class AttentionPack : int64_t
{
    Default  = 0,
    VForFp6P = 1,
};

constexpr int64_t pack_id(AttentionPack pack) { return static_cast<int64_t>(pack); }

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

struct MhaV4Recipe
{
    int64_t q_format;
    int64_t k_format;
    int64_t v_format;
    int64_t v_pack;
    int64_t q_scale_mode;
    int64_t k_scale_mode;
    int64_t v_scale_mode;
};

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

// Sol-Attn kernarg: the same sparse LUT prefix, then the pooled K/V of the approximate branch, the
// selection bitmap, the pooled strides, the pooled SCALES, an optional work table, and the
// block-scaled K's variance pair, 1088 bytes.
//
// The sorted-sparse layout's 0x2D0 work-table slot carries s_kv_range_blocks here instead. A row
// that does not declare kv_range never reads it, and the launcher leaves it 0 for those. Sol-Attn
// cannot reuse the sorted-sparse layout's work table either: that layout puts s_num_wgs and
// s_total_tiles at 0x2E0, where this one starts ptr_mean_k, so its own table sits at 0x410.
//
// The pooled scales from 0x390 exist for the block-granular recipes. Pooling runs along the
// SEQUENCE axis, so whether a descale survives it depends only on whether that descale varies along
// that axis: per-tensor and per-channel ones do not, and mean(x) * descale == mean(x * descale)
// lets the pooled K/V reuse the source descale outright. An E8M0 1x32 scale does vary per token, so
// those recipes have to pool in dequantized space and requantize, which produces scales of its own
// that the approximate pass must read instead of K's or V's. Recipes that do not need them leave
// these slots NULL and keep reading the source scales.
//
// A row that declares `jensen` over a K that is not E8M0, whose K-scale slot is therefore idle,
// reads that slot and its three strides as the optional per-block K variance instead: NULL there
// is the uncorrected pooled pass, non-NULL adds the second-order term to every pooled logit. On an
// FP8 K the variance is of the codes divided by 512 (SOL_ATTN_FP8_VAR_DIVISOR), which the kernel
// undoes; an int8 K's is the same, stored as FP8 E4M3 FN because an integer cannot hold it.
//
// An E8M0 K has no idle scale slot and no single unit for a variance, so its jensen row reads the
// dequantized variance quantized the way mean_k is: codes at 0x420 with mean_k's strides, E8M0
// scales at 0x430 with mean_k_scale's. NULL codes is the uncorrected pass, as above.
//
// A row that declares `sorted` also reads ptr_work_table at 0x410: non-NULL, the grid is 1-D and
// each workgroup takes its (q_tile, head, batch) from the table, heaviest LUT row first; NULL is
// the raster 3-D grid. A row without it declares a 1040-byte kernarg and never sees the slot, and
// a row without the E8M0 variance pair declares at most 1056 bytes and never sees that.
struct __attribute__((packed)) FmhaV4SolKernarg
{
    FmhaV4Kernarg dense;
    ConstPointerSlot ptr_kv_block_indices;
    ConstPointerSlot ptr_lut_start;
    ConstPointerSlot ptr_lut_count;
    ConstPointerSlot ptr_lut_freeze;
    ScalarSlot s_kv_range_blocks;
    ConstPointerSlot ptr_mean_k;
    ConstPointerSlot ptr_mean_v;
    ConstPointerSlot ptr_block_bitmap;
    ScalarSlot s_mean_k_Seqs;
    ScalarSlot s_mean_k_Hs;
    ScalarSlot s_mean_k_Bs;
    ScalarSlot s_mean_v_Seqs;
    ScalarSlot s_mean_v_Hs;
    ScalarSlot s_mean_v_Bs;
    ScalarSlot s_num_kv_blocks;
    ScalarSlot s_bitmap_Ds;
    ConstPointerSlot ptr_mean_k_scale;
    ConstPointerSlot ptr_mean_v_scale;
    ScalarSlot s_mean_k_scale_Seqs;
    ScalarSlot s_mean_k_scale_Hs;
    ScalarSlot s_mean_k_scale_Bs;
    ScalarSlot s_mean_v_scale_Seqs;
    ScalarSlot s_mean_v_scale_Hs;
    ScalarSlot s_mean_v_scale_Bs;
    ConstPointerSlot ptr_work_table;
    ConstPointerSlot ptr_mean_k_var;
    ConstPointerSlot ptr_mean_k_var_scale;
};

static_assert(sizeof(FmhaV4SolKernarg) == 1088,
              "MHA v4 Sol-Attn kernarg ABI must remain 1088 bytes");
static_assert(offsetof(FmhaV4SolKernarg, ptr_work_table) == 0x410);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_k_var) == 0x420);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_k_var_scale) == 0x430);
static_assert(offsetof(FmhaV4SolKernarg, ptr_kv_block_indices) == 0x290);
static_assert(offsetof(FmhaV4SolKernarg, s_kv_range_blocks) == 0x2D0);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_k) == 0x2E0);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_v) == 0x2F0);
static_assert(offsetof(FmhaV4SolKernarg, ptr_block_bitmap) == 0x300);
static_assert(offsetof(FmhaV4SolKernarg, s_mean_k_Seqs) == 0x310);
static_assert(offsetof(FmhaV4SolKernarg, s_mean_v_Seqs) == 0x340);
static_assert(offsetof(FmhaV4SolKernarg, s_num_kv_blocks) == 0x370);
static_assert(offsetof(FmhaV4SolKernarg, s_bitmap_Ds) == 0x380);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_k_scale) == 0x390);
static_assert(offsetof(FmhaV4SolKernarg, ptr_mean_v_scale) == 0x3A0);
static_assert(offsetof(FmhaV4SolKernarg, s_mean_k_scale_Seqs) == 0x3B0);
static_assert(offsetof(FmhaV4SolKernarg, s_mean_v_scale_Seqs) == 0x3E0);

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

// Threads per workgroup for a row. Every kernel in this family gives each wave 32 query rows, so
// the wave count is ts_qo/32 and the workgroup is that many 64-lane waves. Derived rather than
// stored in the manifest so a row cannot declare a tile and a launch width that disagree -- a
// mismatch here does not return wrong numbers, it hangs, because the row's internal
// synchronization waits on waves that were never dispatched.
constexpr int kQueryRowsPerWave = 32;

int workgroup_size_for(const fmha_v4_fwdConfig& cfg)
{
    TORCH_CHECK(cfg.ts_qo % kQueryRowsPerWave == 0,
                "MHA v4 row ",
                cfg.knl_name,
                " has ts_qo=",
                cfg.ts_qo,
                ", which is not a multiple of the ",
                kQueryRowsPerWave,
                " query rows each wave owns");
    return cfg.ts_qo / kQueryRowsPerWave * 64;
}

// A row stages its LUT in a fixed-capacity region -- lut_max in the manifest -- and overrunning
// it reads back whatever follows rather than faulting, so the ceiling is checked here. The bound
// is the shape's, not the row's actual count: a row may legitimately select every block, and
// reading lut_count to find its true maximum would mean a device sync on every launch.
void check_lut_capacity(const fmha_v4_fwdConfig& cfg, int64_t kv_tiles, const char* mode_name)
{
    TORCH_CHECK(cfg.lut_max > 0,
                mode_name,
                " MHA v4 row ",
                cfg.knl_name,
                " declares no lut_max; a block-sparse row must state its LUT capacity");
    TORCH_CHECK(kv_tiles <= cfg.lut_max,
                mode_name,
                " MHA v4 row ",
                cfg.knl_name,
                " holds ",
                cfg.lut_max,
                " LUT entries, but this key length is ",
                kv_tiles,
                " blocks of ",
                cfg.ts_kv,
                " tokens. Split the sequence, or dispatch a geometry with a larger LUT region.");
}

// `ts_qo`/`ts_kv` participate in the key because one arch can ship several tile geometries for the
// SAME recipe and mode (gfx950 has both 256x128 and 64x64 FP8 sparse/Sol-Attn rows), and they are
// not interchangeable: the block mask, LUT and bitmap a caller builds are all in units of ts_kv.
// A block-sparse caller therefore names the geometry it prepared its operands for, so a missing row
// is an error rather than silently whichever row the manifest listed first.
//
// 0 means "any", which is what the dense path passes. Dense builds no mask, so its geometry is an
// implementation detail of the row -- and it genuinely varies (bf16 tiles 256x64 where the
// quantized rows tile 256x128), so there is nothing for a dense caller to name.
const fmha_v4_fwdConfig& find_config(const std::string& arch,
                                     const MhaV4Recipe& recipe,
                                     int64_t mode,
                                     int64_t ts_qo,
                                     int64_t ts_kv)
{
    // cfg_fmha_v4_fwd is an unordered_map, so "first row that matches" is only well defined when at
    // most one row can match. That holds for dense, but gfx950 now has two block-sparse FP8 rows
    // per mode, so leaving the geometry unnamed there would pick a tile by hash order.
    TORCH_CHECK(mode == 0 || (ts_qo != 0 && ts_kv != 0),
                "block-sparse MHA v4 (mode=",
                mode,
                ") requires an explicit q_tile/kv_tile: several tile geometries can serve one "
                "recipe and the LUT, bitmap and pooled K/V are all in units of kv_tile");
    for(const auto& entry : cfg_fmha_v4_fwd)
    {
        const auto& cfg = entry.second;
        if(cfg.arch == arch && cfg.q_format == recipe.q_format &&
           cfg.k_format == recipe.k_format && cfg.v_format == recipe.v_format &&
           cfg.v_pack == recipe.v_pack && cfg.q_scale_mode == recipe.q_scale_mode &&
           cfg.k_scale_mode == recipe.k_scale_mode && cfg.v_scale_mode == recipe.v_scale_mode &&
           cfg.o_format == format_id(AttentionFormat::Bf16) &&
           cfg.o_scale_mode == scale_mode_id(AttentionScaleMode::None) && cfg.hdim_q == kHeadDim &&
           cfg.hdim_v == kHeadDim && cfg.mask == 0 && cfg.mode == mode &&
           (ts_qo == 0 || cfg.ts_qo == ts_qo) && (ts_kv == 0 || cfg.ts_kv == ts_kv))
            return cfg;
    }
    TORCH_CHECK(false,
                "no MHA v4 kernel for arch=",
                arch,
                ", q_format=",
                recipe.q_format,
                ", k_format=",
                recipe.k_format,
                ", v_format=",
                recipe.v_format,
                ", v_pack=",
                recipe.v_pack,
                ", q_scale_mode=",
                recipe.q_scale_mode,
                ", k_scale_mode=",
                recipe.k_scale_mode,
                ", v_scale_mode=",
                recipe.v_scale_mode,
                ", output=BF16, head_dim=128, mode=",
                mode,
                " (0=dense, 1=sorted-sparse, 2=sol-attn), tile=",
                ts_qo,
                "x",
                ts_kv);
}

// E8M0 scale gathers are unguarded global loads: they address every row of the tile they are
// running, plus MXFP4 K's two-tile producer lead, so the final tiles read past the logical rows.
constexpr int64_t kMxScaleBlocksPerRow      = 4;
constexpr int64_t kQueryScaleTileRows       = 256;
constexpr int64_t kKvScaleTileRows          = 128;
constexpr int64_t kKvScaleLookaheadRows     = 2 * kKvScaleTileRows;
constexpr int64_t kKvScaleTrailingDwordSlack = 4;
// MXFP4 V scales are gathered two 512-byte tiles ahead of the tile being run, so the last tiles
// address bytes past the final one whatever the sequence length. Mirrors FP4_V_SCALE_SLACK_BYTES.
constexpr int64_t kMxFp4VScaleSlackBytes = 2 * 512;

void check_scale_backing_storage(const at::Tensor& descale,
                                 int64_t sequence,
                                 int64_t heads,
                                 int64_t tile_rows,
                                 int64_t lookahead_rows,
                                 int64_t trailing_slack,
                                 const char* name)
{
    const int64_t padded = ((sequence + tile_rows - 1) / tile_rows) * tile_rows + lookahead_rows;
    const int64_t required =
        descale.numel() + (padded - sequence) * heads * kMxScaleBlocksPerRow + trailing_slack;
    const int64_t backed = static_cast<int64_t>(descale.storage().nbytes()) -
                           descale.storage_offset() * descale.element_size();
    TORCH_CHECK(backed >= required,
                "MX ",
                name,
                " descale needs ",
                required,
                " mapped bytes so the kernel's speculative tile gather stays in bounds, but only ",
                backed,
                " are backed; allocate it with the aiter.ops.mha_v4_quant producers, which reserve "
                "zeroed slack");
}

// Every stride and extent below occupies a 32-bit kernarg slot. Truncating one would not fault; it
// would silently address the wrong rows, so the launcher refuses the shape instead.
uint32_t fit_u32(int64_t value, const char* name)
{
    TORCH_CHECK(value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "MHA v4 ",
                name,
                " is ",
                value,
                ", which does not fit the 32-bit kernarg slot; this shape is too large for the "
                "current kernel ABI");
    return static_cast<uint32_t>(value);
}

uint32_t byte_stride(const at::Tensor& tensor, int64_t dim, const char* name)
{
    return fit_u32(tensor.stride(dim) * tensor.element_size(), name);
}

void set_descale_strides(const at::Tensor& tensor,
                         int head_dimension,
                         uint32_t& batch_stride,
                         uint32_t& head_stride,
                         const char* name)
{
    if(tensor.dim() >= 2)
    {
        batch_stride = byte_stride(tensor, 0, name);
        head_stride  = byte_stride(tensor, head_dimension, name);
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
// Sol-Attn sorts on lut_count in this many levels of the KV length rather than on the count
// itself, raster order within a level. Its ordinary tiles spread over a range of counts, and an
// exact sort scatters neighbouring query tiles across heads: on 64x64 rows over a 42k-token
// sequence that lost the shared K/V reads and cost 15-38%. Coarse levels still start an all-exact
// (sink) row first, which is the straggler the table is for; 2 to 8 levels measured alike.
constexpr int64_t kSolSortLevels = 4;

// Work-table order. LPT is the global longest-first order the table was built for. HEAD_LOCAL
// keeps heads in raster order and sorts longest-first within each one, so the work items in
// flight share a head's K/V in L2 while every head still starts its heavy tiles first. RASTER is
// the identity. AITER_MHA_V4_WORK_ORDER=lpt|head|raster selects it per call.
enum class WorkOrder : int32_t
{
    Lpt       = 0,
    HeadLocal = 1,
    Raster    = 2,
};

WorkOrder work_order()
{
    const char* value = std::getenv("AITER_MHA_V4_WORK_ORDER");
    if(value == nullptr || value[0] == '\0' || std::strcmp(value, "lpt") == 0)
        return WorkOrder::Lpt;
    if(std::strcmp(value, "head") == 0)
        return WorkOrder::HeadLocal;
    if(std::strcmp(value, "raster") == 0)
        return WorkOrder::Raster;
    TORCH_CHECK(false, "AITER_MHA_V4_WORK_ORDER must be lpt, head or raster, got ", value);
    return WorkOrder::Lpt;
}

// Order and pack in a single launch, by counting each entry's rank rather than moving entries past
// each other. The key carries the whole ordering: the count (divided by bucket, 1 for an exact
// order) is complemented into the high half so longer LUTs come first, and the slot index sits in
// the low half, which breaks ties toward raster order and so makes uniform counts come out as the
// identity permutation.
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
                                                const int32_t nhead,
                                                const int32_t bucket,
                                                const int32_t order)
{
    __shared__ uint64_t keys[kWorkTableFusedMax];

    for(int32_t slot = threadIdx.x; slot < total; slot += blockDim.x)
    {
        const uint32_t level = static_cast<uint32_t>(lut_count[slot] / bucket);
        if(order == static_cast<int32_t>(WorkOrder::HeadLocal))
        {
            // (batch, head) in the top 16 bits, the complemented level below it; batch < 256 and
            // heads < 256 keep the group in range, and a LUT is never longer than 65536.
            const uint64_t group = static_cast<uint32_t>(slot / q_tiles);
            const uint64_t rank  = 0xFFFFu - (level < 0xFFFFu ? level : 0xFFFFu);
            keys[slot] = (group << 48) | (rank << 32) | static_cast<uint32_t>(slot);
        }
        else if(order == static_cast<int32_t>(WorkOrder::Raster))
            keys[slot] = static_cast<uint32_t>(slot);
        else
            keys[slot] = (static_cast<uint64_t>(~level) << 32) |
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
build_sorted_work_table(const at::Tensor& lut_count,
                        int64_t batch,
                        int64_t nhead,
                        int64_t q_tiles,
                        int64_t bucket = 1)
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
    const WorkOrder order    = work_order();
    // Within one head the coarse levels only existed to stop the sort from scattering heads.
    if(order == WorkOrder::HeadLocal)
        bucket = 1;

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
            static_cast<int32_t>(nhead),
            static_cast<int32_t>(bucket),
            static_cast<int32_t>(order));
        return table;
    }

    // Past the LDS staging limit, defer the sort to ATen and pack separately. A stable sort keeps
    // the tie behaviour identical to the fused path.
    at::Tensor permutation;
    if(order == WorkOrder::Raster)
        permutation = at::arange(total, flat.options().dtype(at::kLong));
    else if(order == WorkOrder::HeadLocal)
    {
        const auto group = at::floor_divide(at::arange(total, flat.options().dtype(at::kLong)), q_tiles);
        const auto key   = group * 65536 + (65535 - flat.to(at::kLong).clamp_max(65535));
        permutation      = at::argsort(key, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    }
    else
        permutation = at::argsort(bucket > 1 ? at::floor_divide(flat, bucket) : flat,
                                  /*stable=*/true,
                                  /*dim=*/0,
                                  /*descending=*/true);
    constexpr int32_t block_size = 256;
    const dim3 grid(static_cast<uint32_t>((total + block_size - 1) / block_size));
    pack_work_table_kernel<<<grid, dim3(block_size), 0, stream>>>(table.data_ptr<int32_t>(),
                                                                  permutation.data_ptr<int64_t>(),
                                                                  static_cast<int32_t>(total),
                                                                  static_cast<int32_t>(q_tiles),
                                                                  static_cast<int32_t>(nhead));
    return table;
}

// LUT contents are device data, so the launcher cannot check them without a synchronization. This
// is therefore opt-in: off, an out-of-range index reaches the ASM and faults with only a raw address
// to go on; on, it fails at the launcher with the offending condition named. A row with
// lut_count == 0 is not an error in either mode: sparse skips it and writes zeros for that query
// tile, and Sol-Attn falls back to the pooled-only softmax over every block.
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
                            const MhaV4Recipe& recipe,
                            int64_t seqlen_q,
                            int64_t seqlen_k,
                            int64_t nhead_q,
                            int64_t gqa_ratio,
                            double softmax_scale)
{
    const bool bf16_qk = recipe.q_format == format_id(AttentionFormat::Bf16);
    const bool bf16_v  = recipe.v_format == format_id(AttentionFormat::Bf16);

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
    args.s_seq_len.value     = fit_u32(seqlen_q, "query length");
    args.s_Seqs.value        = byte_stride(q, 1, "Q sequence stride");
    args.s_Ts.value          = fit_u32(cfg.ts_qo * q.stride(1) * q.element_size(), "Q tile stride");
    args.s_Hs.value          = byte_stride(q, 2, "Q head stride");
    args.s_Bs.value          = byte_stride(q, 0, "Q batch stride");
    args.s_gqa.value         = gqa_ratio;
    args.s_k_Seqs.value      = byte_stride(k, 1, "K sequence stride");
    args.s_k_Hs.value        = byte_stride(k, 2, "K head stride");
    args.s_k_Bs.value        = byte_stride(k, 0, "K batch stride");
    args.s_opt.value         = 5;
    // LSE off unless a caller asks: set_lse_kernarg below overwrites these three slots. A row
    // whose code object has no LSE store ignores all of them, which is why asking is checked
    // against the manifest rather than against the pointer being non-null.
    args.s_lse.value         = 0;
    args.s_kv_seq_len.value  = fit_u32(seqlen_k, "key length");
    args.s_qk_head_dim.value = kHeadDim;
    args.s_v_head_dim.value  = kHeadDim;
    args.s_q_head_num.value  = nhead_q;
    args.s_v_Seqs.value      = byte_stride(v, 1, "V sequence stride");
    args.s_v_Hs.value        = byte_stride(v, 2, "V head stride");
    args.s_v_Bs.value        = byte_stride(v, 0, "V batch stride");
    args.s_o_Seqs.value      = byte_stride(out, 1, "output sequence stride");
    args.s_o_Hs.value        = byte_stride(out, 2, "output head stride");
    args.s_o_Bs.value        = byte_stride(out, 0, "output batch stride");

    if(!bf16_qk)
    {
        set_descale_strides(q_descale,
                            q_descale.dim() >= 3 ? 2 : 1,
                            args.s_descale_q_Bs.value,
                            args.s_descale_q_Hs.value,
                            "Q descale stride");
        set_descale_strides(k_descale,
                            k_descale.dim() >= 3 ? 2 : 1,
                            args.s_descale_k_Bs.value,
                            args.s_descale_k_Hs.value,
                            "K descale stride");
    }
    if(!bf16_v)
    {
        set_descale_strides(
            v_descale, 1, args.s_descale_v_Bs.value, args.s_descale_v_Hs.value, "V descale stride");
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

// Point the kernel's LSE slots at `lse`, or leave the launch writing no LSE when it is nullopt.
//
// The kernel writes row (b, h) as seqlen_q unit-stride floats at ptr_lse + (b * nhead_q + h) *
// s_lse_Hs, and writes nothing past seqlen_q. So the rows must be unit-stride float32 and the batch
// stride is not independently settable: a tensor whose batch stride is not nhead_q * head stride
// would have its heads written over each other. A contiguous [batch, nhead_q, seqlen_q] tensor
// satisfies both.
void set_lse_kernarg(FmhaV4Kernarg& args,
                     const std::optional<at::Tensor>& lse,
                     const fmha_v4_fwdConfig& cfg,
                     const at::Tensor& q,
                     int64_t batch,
                     int64_t nhead_q,
                     int64_t seqlen_q,
                     const char* mode_name)
{
    if(!lse.has_value())
    {
        return;
    }
    const at::Tensor& t = lse.value();
    TORCH_CHECK(cfg.lse != 0,
                mode_name,
                " MHA v4 row ",
                cfg.knl_name,
                " has no LSE store in its code object, so return_lse would leave the buffer as "
                "allocated -- which reads as a log-sum-exp of whatever was in it. Use a row whose "
                "manifest lse column is 1.");
    TORCH_CHECK(t.is_cuda() && t.get_device() == q.get_device(),
                mode_name,
                " MHA v4 LSE must be a GPU tensor on the same device as Q");
    TORCH_CHECK(t.scalar_type() == at::kFloat, mode_name, " MHA v4 LSE must be float32");
    TORCH_CHECK(t.dim() == 3 && t.size(0) == batch && t.size(1) == nhead_q &&
                    t.size(2) == seqlen_q,
                mode_name,
                " MHA v4 LSE must be [batch, nhead_q, seqlen_q] = [",
                batch,
                ", ",
                nhead_q,
                ", ",
                seqlen_q,
                "], got ",
                t.sizes());
    TORCH_CHECK(t.stride(2) == 1, mode_name, " MHA v4 LSE rows must be unit-stride");
    TORCH_CHECK(t.stride(0) == nhead_q * t.stride(1),
                mode_name,
                " MHA v4 LSE batch stride must be nhead_q * head stride; the kernel derives the "
                "batch offset from the head stride and cannot honour an independent one");

    args.ptr_lse.value  = t.data_ptr();
    args.s_lse.value    = 1;
    args.s_lse_Hs.value = byte_stride(t, 1, "LSE head stride");
}

PackedMhaV4Shapes validate_packed_mha_v4(const at::Tensor& q,
                                         const at::Tensor& k,
                                         const at::Tensor& v,
                                         const at::Tensor& q_descale,
                                         const at::Tensor& k_descale,
                                         const at::Tensor& v_descale,
                                         const at::Tensor& out,
                                         const MhaV4Recipe& recipe)
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
    TORCH_CHECK(recipe.q_format == recipe.k_format,
                "MHA v4 currently requires matching Q/K formats");
    check_format_tensor(q, recipe.q_format, "Q");
    check_format_tensor(k, recipe.k_format, "K");
    check_format_tensor(v, recipe.v_format, "V");
    TORCH_CHECK(
        recipe.v_pack == pack_id(AttentionPack::Default) ||
            (recipe.v_pack == pack_id(AttentionPack::VForFp6P) &&
             (recipe.v_format == format_id(AttentionFormat::Fp6E2M3) ||
              recipe.v_format == format_id(AttentionFormat::Fp4E2M1))),
        "unsupported MHA v4 V pack for format: v_pack=",
        recipe.v_pack,
        ", v_format=",
        recipe.v_format);
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1,
                "Q, K, V, and out must have contiguous last dimensions");

    PackedMhaV4Shapes shapes{};
    shapes.batch               = q.size(0);
    shapes.seqlen_q            = q.size(1);
    shapes.nhead_q             = q.size(2);
    shapes.seqlen_k            = k.size(1);
    shapes.nhead_k             = k.size(2);
    const int64_t packed_width =
        recipe.q_format == format_id(AttentionFormat::Fp6E2M3)   ? 96
        : recipe.q_format == format_id(AttentionFormat::Fp4E2M1) ? 64
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
    if(recipe.q_format == format_id(AttentionFormat::Fp4E2M1))
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

    const bool mx_qk_format = recipe.q_format == format_id(AttentionFormat::Fp6E2M3) ||
                              recipe.q_format == format_id(AttentionFormat::Fp4E2M1);
    const bool bf16_qk = recipe.q_format == format_id(AttentionFormat::Bf16);
    const bool bf16_v  = recipe.v_format == format_id(AttentionFormat::Bf16);
    const bool e8m0_qk_scales =
        recipe.q_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32) &&
        recipe.k_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32);
    if(bf16_qk)
    {
        TORCH_CHECK(recipe.q_scale_mode == scale_mode_id(AttentionScaleMode::None) &&
                recipe.k_scale_mode == scale_mode_id(AttentionScaleMode::None),
                    "BF16 Q/K must use NONE scale modes");
        TORCH_CHECK((bf16_v &&
                     recipe.v_scale_mode == scale_mode_id(AttentionScaleMode::None)) ||
                        (!bf16_v &&
                         recipe.v_scale_mode == scale_mode_id(AttentionScaleMode::F32PerTensor)),
                    "BF16 Q/K requires NONE scale mode for BF16 V or F32_PER_TENSOR for FP8 V");
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
        check_scale_backing_storage(q_descale,
                                    shapes.seqlen_q,
                                    shapes.nhead_q,
                                    kQueryScaleTileRows,
                                    0,
                                    0,
                                    "Q");
        if(recipe.k_format == format_id(AttentionFormat::Fp4E2M1))
        {
            check_scale_backing_storage(k_descale,
                                        shapes.seqlen_k,
                                        shapes.nhead_k,
                                        kKvScaleTileRows,
                                        kKvScaleLookaheadRows,
                                        kKvScaleTrailingDwordSlack,
                                        "K");
        }
    }
    else
    {
        TORCH_CHECK(q_descale.scalar_type() == at::ScalarType::Float &&
                        k_descale.scalar_type() == at::ScalarType::Float,
                    "INT8/FP8 Q/K descales must be float32 tensors");
        TORCH_CHECK(q_descale.numel() == 1 && k_descale.numel() == 1,
                    "INT8/FP8 Q/K descales must be scalar tensors");
    }
    const bool mx_v = recipe.v_format == format_id(AttentionFormat::Fp6E2M3) ||
                      recipe.v_format == format_id(AttentionFormat::Fp4E2M1);
    if(bf16_qk && bf16_v)
    {
        // Raw BF16 operands do not use descale tensors.
    }
    else if(bf16_qk)
    {
        TORCH_CHECK(v_descale.scalar_type() == at::ScalarType::Float && v_descale.numel() == 1,
                    "BF16/FP8 V descale must be a scalar float32 tensor");
    }
    else if(mx_v)
    {
        const int64_t tiles = (shapes.seqlen_k + 127) / 128;
        TORCH_CHECK(recipe.v_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32) &&
                v_descale.scalar_type() == at::ScalarType::Byte,
                    "MX V descale must use uint8 E8M0 per-1x32 scales");
        TORCH_CHECK(v_descale.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.nhead_k, tiles * 512}),
                    "MX V descale must have shape [batch, key_heads, tiles * 512]");
        if(recipe.v_format == format_id(AttentionFormat::Fp4E2M1))
        {
            const int64_t required = v_descale.numel() + kMxFp4VScaleSlackBytes;
            const int64_t backed   = static_cast<int64_t>(v_descale.storage().nbytes()) -
                                   v_descale.storage_offset() * v_descale.element_size();
            TORCH_CHECK(backed >= required,
                        "MX V descale needs ",
                        required,
                        " mapped bytes so the kernel's speculative tile gather stays in bounds, "
                        "but only ",
                        backed,
                        " are backed; allocate it with the aiter.ops.mha_v4_quant producers, "
                        "which reserve zeroed slack");
        }
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
                 int64_t v_pack,
                 int64_t q_scale_mode,
                 int64_t k_scale_mode,
                 int64_t v_scale_mode,
                 double softmax_scale,
                 std::optional<at::Tensor> seqlens_k,
                 std::optional<at::Tensor> lse)
{
    const MhaV4Recipe recipe{q_format,
                             k_format,
                             v_format,
                             v_pack,
                             q_scale_mode,
                             k_scale_mode,
                             v_scale_mode};
    const auto shapes = validate_packed_mha_v4(q,
                                               k,
                                               v,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               out,
                                               recipe);

    // Before any device query or launch: get_gpu_arch() reads whichever device is current, and
    // every launch below inherits the current device's stream.
    const HipDeviceGuard device_guard{q.get_device()};

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(arch, recipe, /*mode=*/0, /*ts_qo=*/0, /*ts_kv=*/0);

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
                           recipe,
                           shapes.seqlen_q,
                           shapes.seqlen_k,
                           shapes.nhead_q,
                           shapes.gqa_ratio,
                           softmax_scale);
    set_lse_kernarg(
        args, lse, cfg, q, shapes.batch, shapes.nhead_q, shapes.seqlen_q, "dense");

    // Per-batch key lengths are optional: the kernels read this slot only when it is non-null, so a
    // dense launch leaves it zero rather than selecting a different code object.
    if(seqlens_k.has_value())
    {
        TORCH_CHECK(seqlens_k->is_cuda() && seqlens_k->device() == q.device(),
                    "MHA v4 seqlens_k must be a GPU tensor on the same device as Q");
        TORCH_CHECK(seqlens_k->scalar_type() == at::kInt,
                    "MHA v4 seqlens_k must be int32, got ",
                    seqlens_k->scalar_type());
        TORCH_CHECK(seqlens_k->is_contiguous(), "MHA v4 seqlens_k must be contiguous");
        TORCH_CHECK(seqlens_k->numel() >= shapes.batch,
                    "MHA v4 seqlens_k needs one entry per batch: got ",
                    seqlens_k->numel(),
                    " for batch ",
                    shapes.batch);
        // Contents stay the caller's contract, as they do for cu_seqlens on the varlen path:
        // they live on the device, so bounding them here would synchronize every launch.
        args.ptr_kseq.value = seqlens_k->data_ptr();
    }

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel                = kernels.get_or_create(
        cache_key, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    size_t arg_size          = sizeof(args);
    const int gdx            = (shapes.seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int gdy            = shapes.nhead_q;
    const int gdz            = shapes.batch;
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, gdx, gdy, gdz, workgroup_size_for(cfg), 1, 1, stream});
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
                        int64_t v_pack,
                        int64_t q_scale_mode,
                        int64_t k_scale_mode,
                        int64_t v_scale_mode,
                        double softmax_scale,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count,
                        int64_t q_tile,
                        int64_t kv_tile,
                        std::optional<at::Tensor> lse)
{
    const MhaV4Recipe recipe{q_format,
                             k_format,
                             v_format,
                             v_pack,
                             q_scale_mode,
                             k_scale_mode,
                             v_scale_mode};
    const auto shapes = validate_packed_mha_v4(q,
                                               k,
                                               v,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               out,
                                               recipe);

    // Before any device query or launch. build_sorted_work_table() below launches raw HIP kernels,
    // which take the current device and its stream rather than Q's, so an unguarded call on a
    // non-current device faults writing the work table.
    const HipDeviceGuard device_guard{q.get_device()};

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(arch, recipe, /*mode=*/1, q_tile, kv_tile);
    TORCH_CHECK(cfg.ragged_kv != 0 || shapes.seqlen_k % cfg.ts_kv == 0,
                "sorted-sparse MHA v4 row ",
                cfg.knl_name,
                " requires key length padded to a multiple of ",
                cfg.ts_kv);

    const int64_t q_tiles  = (shapes.seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int64_t kv_tiles = (shapes.seqlen_k + cfg.ts_kv - 1) / cfg.ts_kv;
    const int64_t lut_rows = shapes.batch * shapes.nhead_q * q_tiles;
    check_lut_capacity(cfg, kv_tiles, "sorted-sparse");
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
                           recipe,
                           shapes.seqlen_q,
                           shapes.seqlen_k,
                           shapes.nhead_q,
                           shapes.gqa_ratio,
                           softmax_scale);
    set_lse_kernarg(
        args.dense, lse, cfg, q, shapes.batch, shapes.nhead_q, shapes.seqlen_q, "sorted-sparse");
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
    kernel.launch_kernel({&args, &arg_size, static_cast<int>(lut_rows), 1, 1, workgroup_size_for(cfg), 1, 1, stream});
}

void fmha_v4_fwd_sol(const at::Tensor& q,
                     const at::Tensor& k,
                     const at::Tensor& v,
                     const at::Tensor& q_descale,
                     const at::Tensor& k_descale,
                     const at::Tensor& v_descale,
                     at::Tensor out,
                     int64_t q_format,
                     int64_t k_format,
                     int64_t v_format,
                     int64_t v_pack,
                     int64_t q_scale_mode,
                     int64_t k_scale_mode,
                     int64_t v_scale_mode,
                     double softmax_scale,
                     const at::Tensor& kv_block_indices,
                     const at::Tensor& lut_start,
                     const at::Tensor& lut_count,
                     const at::Tensor& mean_k,
                     const at::Tensor& mean_v,
                     const at::Tensor& block_bitmap,
                     const std::optional<at::Tensor>& mean_k_scale,
                     const std::optional<at::Tensor>& mean_v_scale,
                     int64_t q_tile,
                     int64_t kv_tile,
                     std::optional<at::Tensor> lse,
                     int64_t kv_range_tokens,
                     const std::optional<at::Tensor>& mean_k_var,
                     int64_t sorted_dispatch,
                     const std::optional<at::Tensor>& mean_k_var_scale)
{
    const MhaV4Recipe recipe{q_format,
                             k_format,
                             v_format,
                             v_pack,
                             q_scale_mode,
                             k_scale_mode,
                             v_scale_mode};
    const auto shapes = validate_packed_mha_v4(q,
                                               k,
                                               v,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               out,
                                               recipe);

    const HipDeviceGuard device_guard{q.get_device()};

    const auto arch = get_gpu_arch();
    const auto& cfg = find_config(arch, recipe, /*mode=*/2, q_tile, kv_tile);
    // Matches the sorted-sparse sibling, whose LUT machinery Sol-Attn reuses verbatim for its exact
    // pass. A short last block is only ever computed there: sol_prepare() forces it exact,
    // since the pooled xts_kv factor only holds for a full block.
    TORCH_CHECK(cfg.ragged_kv != 0 || shapes.seqlen_k % cfg.ts_kv == 0,
                "Sol-Attn MHA v4 row ",
                cfg.knl_name,
                " requires key length padded to a multiple of ",
                cfg.ts_kv);

    const int64_t q_tiles       = (shapes.seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
    const int64_t kv_tiles      = (shapes.seqlen_k + cfg.ts_kv - 1) / cfg.ts_kv;
    const int64_t lut_rows      = shapes.batch * shapes.nhead_q * q_tiles;
    check_lut_capacity(cfg, kv_tiles, "Sol-Attn");
    // Ranges are laid end to end and each has to start on a 32-block word of the selection bitmap.
    const int64_t kv_range_quantum = 32 * cfg.ts_kv;
    TORCH_CHECK(kv_range_tokens >= 0, "kv_range_tokens must be non-negative");
    TORCH_CHECK(kv_range_tokens == 0 || cfg.kv_range != 0,
                "Sol-Attn row ",
                cfg.knl_name,
                " has no in-kernel KV range reset; pass kv_range_tokens=0");
    TORCH_CHECK(kv_range_tokens % kv_range_quantum == 0,
                "kv_range_tokens must be a multiple of ",
                kv_range_quantum,
                " (32 blocks of ",
                cfg.ts_kv,
                " keys); got ",
                kv_range_tokens);
    // The approximate pass sweeps pooled blocks the way the exact pass sweeps tokens, so one of its
    // tiles covers cfg.ts_kv BLOCKS and needs that many mask bits: ts_kv / 32 words, read as one
    // aligned load. Everything the pass reads per tile is grouped and padded to that stride.
    const int64_t blocks_per_tile = cfg.ts_kv;
    const int64_t bitmap_groups   = (kv_tiles + blocks_per_tile - 1) / blocks_per_tile;
    const int64_t bitmap_ds       = blocks_per_tile / 32 * bitmap_groups;

    TORCH_CHECK(kv_block_indices.is_cuda() && lut_start.is_cuda() && lut_count.is_cuda(),
                "LUT tensors must be GPU tensors");
    TORCH_CHECK(kv_block_indices.device() == q.device() && lut_start.device() == q.device() &&
                    lut_count.device() == q.device(),
                "LUT tensors must be on the same GPU as Q");
    TORCH_CHECK(kv_block_indices.scalar_type() == at::ScalarType::Int &&
                    lut_start.scalar_type() == at::ScalarType::Int &&
                    lut_count.scalar_type() == at::ScalarType::Int,
                "LUT tensors must be int32");
    TORCH_CHECK(kv_block_indices.dim() == 1 && lut_start.dim() == 1 && lut_count.dim() == 1,
                "LUT tensors must be 1-D");
    TORCH_CHECK(lut_start.numel() == lut_rows && lut_count.numel() == lut_rows,
                "lut_start and lut_count must have one entry per (batch, head, query tile); "
                "expected ",
                lut_rows);
    TORCH_CHECK(kv_block_indices.numel() >= 1,
                "kv_block_indices must be non-empty; the sparse kernels dereference the row base "
                "even for a row that selects no KV block");

    // The pooled tensors are K and V with seqlen -> num_kv_blocks, in the SOURCE quantized dtype,
    // so the approximate pass reuses q/k/v_descale unchanged. That identity is what keeps the
    // per-tensor recipes free; a block-granular scale mode would need its own pooled scales, which
    // is why only per-tensor rows carry a mode-2 manifest entry today.
    TORCH_CHECK(mean_k.is_cuda() && mean_v.is_cuda() && block_bitmap.is_cuda(),
                "Sol-Attn pooled tensors and bitmap must be GPU tensors");
    TORCH_CHECK(mean_k.device() == q.device() && mean_v.device() == q.device() &&
                    block_bitmap.device() == q.device(),
                "Sol-Attn pooled tensors and bitmap must be on the same GPU as Q");
    TORCH_CHECK(mean_k.scalar_type() == k.scalar_type(),
                "mean_k must have K's dtype so the approximate pass can reuse k_descale");
    TORCH_CHECK(mean_v.scalar_type() == v.scalar_type(),
                "mean_v must have V's dtype so the approximate pass can reuse v_descale");
    TORCH_CHECK(mean_k.sizes() == torch::IntArrayRef({shapes.batch,
                                                      kv_tiles,
                                                      shapes.nhead_k,
                                                      k.size(3)}),
                "mean_k must have shape [batch, num_kv_blocks, key_heads, K packed width]");
    TORCH_CHECK(mean_v.sizes() == torch::IntArrayRef({shapes.batch,
                                                      kv_tiles,
                                                      shapes.nhead_k,
                                                      v.size(3)}),
                "mean_v must have shape [batch, num_kv_blocks, key_heads, V head dim]");
    TORCH_CHECK(mean_k.stride(3) == 1 && mean_v.stride(3) == 1,
                "Sol-Attn pooled tensors must have contiguous last dimensions");
    // One aligned load per approximate tile, at byte offset (ts_kv / 8) * tile, so a row length
    // that is not a whole number of groups would misalign every tile after the first.
    TORCH_CHECK(block_bitmap.element_size() == 4 &&
                    (block_bitmap.scalar_type() == at::ScalarType::UInt32 ||
                     block_bitmap.scalar_type() == at::ScalarType::Int),
                "block_bitmap must be a 32-bit integer tensor");
    TORCH_CHECK(block_bitmap.dim() == 2 &&
                    block_bitmap.sizes() == torch::IntArrayRef({lut_rows, bitmap_ds}),
                "block_bitmap must have shape [batch * query_heads * query_tiles, "
                "(ts_kv / 32) * ceil(num_kv_blocks / ts_kv)]; expected [",
                lut_rows,
                ", ",
                bitmap_ds,
                "]");
    TORCH_CHECK(block_bitmap.is_contiguous(), "block_bitmap must be contiguous");

    // Whether an operand needs a pooled scale is decided by its scale mode, not by the caller: an
    // E8M0 1x32 scale varies along the axis pooling reduces, so that operand had to be pooled in
    // dequantized space and requantized, and the approximate pass must read the scale that came out
    // of that rather than the source one. Anything that does not vary along the sequence axis
    // survives pooling untouched, so passing a pooled scale for it would describe a read the kernel
    // never performs.
    const bool k_needs_pooled_scale =
        k_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32);
    const bool v_needs_pooled_scale =
        v_scale_mode == scale_mode_id(AttentionScaleMode::E8M0Per1x32);
    TORCH_CHECK(mean_k_scale.has_value() == k_needs_pooled_scale,
                "Sol-Attn needs mean_k_scale exactly when K's scale mode is E8M0_PER_1X32; K's is ",
                k_scale_mode);
    TORCH_CHECK(mean_v_scale.has_value() == v_needs_pooled_scale,
                "Sol-Attn needs mean_v_scale exactly when V's scale mode is E8M0_PER_1X32; V's is ",
                v_scale_mode);
    if(k_needs_pooled_scale)
    {
        const auto& ks = mean_k_scale.value();
        TORCH_CHECK(ks.is_cuda() && ks.device() == q.device(),
                    "mean_k_scale must be on the same GPU as Q");
        TORCH_CHECK(ks.scalar_type() == at::ScalarType::Byte,
                    "mean_k_scale must be a uint8 E8M0 tensor");
        // Only num_kv_blocks rows carry meaning, but the height may be padded up to a whole tile.
        // Some recipes read a full tile of scale bytes however short the pooled image is, and an
        // uninitialized E8M0 byte of 0xFF is 2^128, which reaches QK as inf before the bitmap masks
        // the column out. Those have to pass a zero-padded image, so accept either height.
        const int64_t padded_rows = bitmap_groups * blocks_per_tile;
        TORCH_CHECK(ks.dim() == 4 && ks.size(0) == shapes.batch &&
                        ks.size(2) == shapes.nhead_k && ks.size(3) == kHeadDim / 32 &&
                        (ks.size(1) == kv_tiles || ks.size(1) == padded_rows),
                    "mean_k_scale must have shape [batch, rows, key_heads, 4] with rows either "
                    "num_kv_blocks (",
                    kv_tiles,
                    ") or that padded to a whole tile (",
                    padded_rows,
                    "), matching K's own scale image with key_length replaced by num_kv_blocks");
        TORCH_CHECK(ks.stride(3) == 1, "mean_k_scale must have a contiguous last dimension");
    }
    if(v_needs_pooled_scale)
    {
        // The V-scale image is packed, not strided, and carries no stride slots of its own, so all
        // it needs is a base with whole tiles behind it. Hence a size check only. It is packed in
        // V's own 128-token records whatever ts_kv is, over the pooled rows padded to a whole tile.
        const auto& vs             = mean_v_scale.value();
        const int64_t pooled_rows  = bitmap_groups * blocks_per_tile;
        const int64_t pooled_bytes = (pooled_rows + 127) / 128 * 512;
        TORCH_CHECK(vs.is_cuda() && vs.device() == q.device(),
                    "mean_v_scale must be on the same GPU as Q");
        TORCH_CHECK(vs.scalar_type() == at::ScalarType::Byte,
                    "mean_v_scale must be a uint8 E8M0 tensor");
        TORCH_CHECK(vs.sizes() ==
                        torch::IntArrayRef({shapes.batch, shapes.nhead_k, pooled_bytes}),
                    "mean_v_scale must have shape [batch, key_heads, "
                    "ceil(ceil(num_kv_blocks / ts_kv) * ts_kv / 128) * 512], matching V's own "
                    "packed scale image with key_length replaced by num_kv_blocks padded to a "
                    "whole tile; expected ",
                    pooled_bytes,
                    " bytes per head");
        TORCH_CHECK(vs.is_contiguous(), "mean_v_scale must be contiguous");
    }
    TORCH_CHECK(!mean_k_var_scale.has_value() || mean_k_var.has_value(),
                "mean_k_var_scale is mean_k_var's scale; it needs mean_k_var");
    if(mean_k_var.has_value())
    {
        const auto& kvar = mean_k_var.value();
        TORCH_CHECK(cfg.jensen != 0,
                    "Sol-Attn row ",
                    cfg.knl_name,
                    " has no second-order pooled correction; pass mean_k_var=None");
        TORCH_CHECK(kvar.is_cuda() && kvar.device() == q.device(),
                    "mean_k_var must be on the same GPU as Q");
        TORCH_CHECK(kvar.sizes() == mean_k.sizes(), "mean_k_var must have mean_k's shape");
        if(k_needs_pooled_scale)
        {
            // The kernel reads both halves through mean_k's and mean_k_scale's addressing, so only
            // the two bases are passed.
            TORCH_CHECK(mean_k_var_scale.has_value(),
                        "an E8M0_PER_1X32 K's mean_k_var is block-scaled; pass mean_k_var_scale");
            const auto& vs = mean_k_var_scale.value();
            const auto& ks = mean_k_scale.value();
            TORCH_CHECK(kvar.scalar_type() == mean_k.scalar_type(),
                        "mean_k_var must have mean_k's dtype");
            TORCH_CHECK(kvar.strides() == mean_k.strides(), "mean_k_var must have mean_k's strides");
            TORCH_CHECK(vs.is_cuda() && vs.device() == q.device(),
                        "mean_k_var_scale must be on the same GPU as Q");
            TORCH_CHECK(vs.scalar_type() == at::ScalarType::Byte,
                        "mean_k_var_scale must be a uint8 E8M0 tensor");
            TORCH_CHECK(vs.sizes() == ks.sizes() && vs.strides() == ks.strides(),
                        "mean_k_var_scale must have mean_k_scale's shape and strides");
        }
        else
        {
            // The kernel reads the variance with mean_k's block stride; only its base and
            // head/batch strides are passed separately.
            TORCH_CHECK(!mean_k_var_scale.has_value(),
                        "mean_k_var_scale is for an E8M0_PER_1X32 K; this K's variance has no scale");
            const auto var_type = mean_k.scalar_type() == at::ScalarType::Char
                                      ? at::ScalarType::Float8_e4m3fn
                                      : mean_k.scalar_type();
            TORCH_CHECK(kvar.scalar_type() == var_type,
                        "mean_k_var must have mean_k's dtype, or FP8 E4M3 FN for an int8 K");
            TORCH_CHECK(kvar.stride(3) == 1 && kvar.stride(1) == mean_k.stride(1),
                        "mean_k_var must have a contiguous last dimension and mean_k's block "
                        "stride");
        }
    }

    // -1 sorts wherever the row supports it and the table's packing can address the grid, 0 keeps
    // the raster grid, 1 insists. Rows whose LUT counts sit in one level lose nothing to it: ties
    // keep raster order, so the table is the identity permutation. See kSolSortLevels.
    TORCH_CHECK(sorted_dispatch >= -1 && sorted_dispatch <= 1,
                "sorted_dispatch must be -1 (auto), 0 (raster) or 1 (sorted), got ",
                sorted_dispatch);
    TORCH_CHECK(sorted_dispatch != 1 || cfg.sorted != 0,
                "Sol-Attn MHA v4 kernel ",
                cfg.knl_name,
                " has no sorted dispatch; pass sorted_dispatch=0 or -1");
    const bool packable = shapes.batch < 256 && shapes.nhead_q < 256 && q_tiles < 65536;
    const bool use_sorted =
        cfg.sorted != 0 && (sorted_dispatch == 1 || (sorted_dispatch == -1 && packable));

    const auto kv_idx = kv_block_indices.contiguous();
    const auto start  = lut_start.contiguous();
    const auto count  = lut_count.contiguous();
    if(lut_validation_enabled())
    {
        validate_lut_contents(kv_idx, start, count, lut_rows, kv_tiles);
    }
    at::Tensor work_table;
    if(use_sorted)
    {
        const int64_t bucket = (kv_tiles + kSolSortLevels - 1) / kSolSortLevels;
        work_table = build_sorted_work_table(count, shapes.batch, shapes.nhead_q, q_tiles, bucket);
    }

    FmhaV4SolKernarg args{};
    populate_dense_kernarg(args.dense,
                           q,
                           k,
                           v,
                           q_descale,
                           k_descale,
                           v_descale,
                           out,
                           cfg,
                           recipe,
                           shapes.seqlen_q,
                           shapes.seqlen_k,
                           shapes.nhead_q,
                           shapes.gqa_ratio,
                           softmax_scale);
    // The Sol-Attn LSE is the denominator of the ONE softmax the two branches share, so it counts
    // the pooled proxy columns as well as the exact ones. That is the quantity partial outputs
    // merge by, and it merges exactly, because a merge adds numerators and denominators separately
    // and the proxy sits in both. What it does not carry is which blocks were proxied: that is
    // decided from the blocks a caller hands this launch, so a caller splitting the KV across ranks
    // owns keeping the routing consistent. See the note on mha_v4_sol.
    set_lse_kernarg(args.dense,
                    lse,
                    cfg,
                    q,
                    shapes.batch,
                    shapes.nhead_q,
                    shapes.seqlen_q,
                    "Sol-Attn");
    args.ptr_kv_block_indices.value = kv_idx.data_ptr();
    args.ptr_lut_start.value        = start.data_ptr();
    args.ptr_lut_count.value        = count.data_ptr();
    args.s_kv_range_blocks.value    = static_cast<uint32_t>(kv_range_tokens / cfg.ts_kv);
    args.ptr_mean_k.value           = mean_k.data_ptr();
    args.ptr_mean_v.value           = mean_v.data_ptr();
    args.ptr_block_bitmap.value     = block_bitmap.data_ptr();
    args.s_mean_k_Seqs.value        = byte_stride(mean_k, 1, "pooled K sequence stride");
    args.s_mean_k_Hs.value          = byte_stride(mean_k, 2, "pooled K head stride");
    args.s_mean_k_Bs.value          = byte_stride(mean_k, 0, "pooled K batch stride");
    args.s_mean_v_Seqs.value        = byte_stride(mean_v, 1, "pooled V sequence stride");
    args.s_mean_v_Hs.value          = byte_stride(mean_v, 2, "pooled V head stride");
    args.s_mean_v_Bs.value          = byte_stride(mean_v, 0, "pooled V batch stride");
    args.s_num_kv_blocks.value      = static_cast<uint32_t>(kv_tiles);
    args.s_bitmap_Ds.value          = static_cast<uint32_t>(bitmap_ds);
    // Left NULL for the per-tensor recipes, which is what keeps their scale reads on the source
    // image. s_mean_v_scale_* stay unset on purpose: no recipe reads them, since the packed V-scale
    // image needs only a base pointer.
    if(mean_k_scale.has_value())
    {
        const auto& ks                  = mean_k_scale.value();
        args.ptr_mean_k_scale.value     = ks.data_ptr();
        args.s_mean_k_scale_Seqs.value  = byte_stride(ks, 1, "pooled K scale sequence stride");
        args.s_mean_k_scale_Hs.value    = byte_stride(ks, 2, "pooled K scale head stride");
        args.s_mean_k_scale_Bs.value    = byte_stride(ks, 0, "pooled K scale batch stride");
    }
    if(mean_v_scale.has_value())
    {
        args.ptr_mean_v_scale.value = mean_v_scale.value().data_ptr();
    }
    if(mean_k_var_scale.has_value())
    {
        args.ptr_mean_k_var.value       = mean_k_var.value().data_ptr();
        args.ptr_mean_k_var_scale.value = mean_k_var_scale.value().data_ptr();
    }
    else if(mean_k_var.has_value())
    {
        const auto& kvar                = mean_k_var.value();
        args.ptr_mean_k_scale.value     = kvar.data_ptr();
        args.s_mean_k_scale_Seqs.value  = byte_stride(kvar, 1, "K variance sequence stride");
        args.s_mean_k_scale_Hs.value    = byte_stride(kvar, 2, "K variance head stride");
        args.s_mean_k_scale_Bs.value    = byte_stride(kvar, 0, "K variance batch stride");
    }
    if(use_sorted)
    {
        args.ptr_work_table.value = work_table.data_ptr();
    }

    static SynchronizedCache<std::string, AiterAsmKernel> kernels;
    const std::string cache_key = arch + "|" + cfg.knl_name + "|" + cfg.co_name;
    auto& kernel                = kernels.get_or_create(
        cache_key, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    size_t arg_size          = sizeof(args);
    const int gdx            = static_cast<int>(use_sorted ? lut_rows : q_tiles);
    const int gdy            = use_sorted ? 1 : static_cast<int>(shapes.nhead_q);
    const int gdz            = use_sorted ? 1 : static_cast<int>(shapes.batch);
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    kernel.launch_kernel({&args, &arg_size, gdx, gdy, gdz, workgroup_size_for(cfg), 1, 1, stream});
}

} // namespace torch_itfs
} // namespace aiter