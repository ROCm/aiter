// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 Mega-MHC asm kernels (hsa/gfx1250/mega_mhc, H = 5120, split16).
//
// The kernels take the explicit arguments of the gfx1250 HIP mega_mhc_kernel
// (csrc/kernels/mega_mhc_gfx1250.cu) up to scale_row_major, in a 192-byte
// kernarg record whose first 30 DWORDs are preloaded into SGPRs. They use the
// same packed fn (mega_mhc_shuffle_fn), inputs and outputs as the HIP kernel.
//
//  kind | tile  | grid (workgroups)                          | partials (FP32)
// ------+-------+--------------------------------------------+-----------------
//  ps   |  64   | min(ps_clusters, ceil(T/64)) * 16, persist | 16*T*32 + 16*T
//  ns   | 16/32 | ceil(T/tile) * 16                          | 16*T*32 + 16*T
//
// Each tile has BF16, MXFP8 row-major and MXFP8 M32K4 builds for every
// (HAS_POST, IDENTITY_PRE). MXFP8 writes the BF16 norm, E4M3fn q and E8M0
// scales: [T, H/32] row-major, or the gfx1250 M32K4 GEMM A-scale layout over
// pad32(T) rows with 0x7F padding. The scale layout is fixed per code object.
//
// Every launch: block 128 (4 wave32), workgroup clusters of 16 (one per split),
// split_arg 16, LDS 320 KiB. Partials are indexed by global token, need no
// initialization and are never read back by a later launch; counters,
// hidden_arg and tail_token0 are not read.
// aiter_ctypes_error.h needs aiter_hip_common.h, which aiter_tensor.h includes.
#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"
#include "asm_mega_mhc_configs.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>

namespace {

constexpr int64_t MHC_HIDDEN      = 5120;
constexpr int MHC_SPLITS          = 16;
constexpr int MHC_BLOCK           = 128;
constexpr int MHC_PARTIAL_WORDS   = 32;
constexpr size_t MHC_KERNARG_SIZE = 192;

struct __attribute__((packed)) MegaMhcKernelArgs
{
    const void* residual;
    const void* sublayer;
    const void* post_mix;
    const void* comb_mix;
    const void* pre_mix;
    const void* fn_packed;
    int32_t tokens;
    int32_t split_arg;
    void* residual_out;
    void* post_out;
    void* comb_out;
    void* layer_input;
    void* next_pre;
    const void* hc_scale;
    const void* hc_base;
    const void* norm_weight;
    void* partials;
    void* counters;
    int32_t hidden_arg;
    float rms_eps;
    float pre_eps;
    float sinkhorn_eps;
    float post_mult;
    float norm_eps;
    int32_t sinkhorn_repeat;
    int32_t tail_token0;
    void* quant_out;
    void* quant_scale;
    uint8_t scale_row_major;
    uint8_t _pad[7];
};
static_assert(sizeof(MegaMhcKernelArgs) == MHC_KERNARG_SIZE,
              "mega_mhc asm kernarg record is 192 B");
static_assert(offsetof(MegaMhcKernelArgs, fn_packed) == 40, "fn_packed offset");
static_assert(offsetof(MegaMhcKernelArgs, tokens) == 48, "tokens offset");
static_assert(offsetof(MegaMhcKernelArgs, residual_out) == 56, "residual_out offset");
static_assert(offsetof(MegaMhcKernelArgs, partials) == 120, "partials offset");
static_assert(offsetof(MegaMhcKernelArgs, hidden_arg) == 136, "hidden_arg offset");
static_assert(offsetof(MegaMhcKernelArgs, sinkhorn_repeat) == 160, "sinkhorn_repeat offset");
static_assert(offsetof(MegaMhcKernelArgs, quant_out) == 168, "quant_out offset");
static_assert(offsetof(MegaMhcKernelArgs, scale_row_major) == 184, "scale_row_major offset");

const mega_mhcConfig* find_config(int64_t tile_tokens,
                                  bool mxfp8,
                                  bool scale_row_major,
                                  bool has_post,
                                  bool identity_pre,
                                  int64_t cs)
{
    const std::string arch      = get_gpu_arch();
    const mega_mhcConfig* found = nullptr;
    for(const auto& el : cfg_mega_mhc)
    {
        const auto& cfg = el.second;
        if(cfg.arch != arch || cfg.tile_tokens != tile_tokens || (cfg.out_mxfp8 != 0) != mxfp8 ||
           (mxfp8 && (cfg.scale_row_major != 0) != scale_row_major) ||
           (cfg.has_post != 0) != has_post || (cfg.identity_pre != 0) != identity_pre ||
           cfg.cs != cs)
            continue;
        // Deterministic choice if several manifests register the same shape.
        if(found == nullptr || cfg.knl_name < found->knl_name)
            found = &cfg;
    }
    return found;
}

void check_tensor(const aiter_tensor_t* t,
                  const aiter_tensor_t* ref,
                  AiterDtype dtype,
                  size_t numel,
                  const char* name)
{
    AITER_CHECK(t != nullptr, name, " is required");
    AITER_CHECK(
        t->is_gpu() && t->device_id == ref->device_id, name, " must be on the residual device");
    AITER_CHECK(t->dtype() == dtype && t->is_contiguous() && t->numel() >= numel,
                name,
                " has an invalid dtype, size or layout");
    AITER_CHECK(numel == 0 || reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                name,
                " must be 16-byte aligned");
}

} // namespace

AITER_CTYPES_ERROR_DEF

// 1 when a code object for this shape is registered on the current device's arch.
AITER_CTYPES_DEFINE_ENTRYPOINT(
    mega_mhc_asm_available,
    (int64_t tile_tokens,
     int mxfp8,
     int scale_row_major,
     int has_post,
     int identity_pre,
     int64_t cs,
     hipStream_t stream),
    (tile_tokens, mxfp8, scale_row_major, has_post, identity_pre, cs, stream))
{
    (void)stream;
    return find_config(tile_tokens,
                       mxfp8 != 0,
                       scale_row_major != 0,
                       has_post != 0,
                       identity_pre != 0,
                       cs) != nullptr;
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mega_mhc_asm,
    (aiter_tensor_t * residual_out, // (T, 4, H) BF16; the residual itself when has_post = 0
     aiter_tensor_t* post_out,      // (T, 4, 1) FP32
     aiter_tensor_t* comb_out,      // (T, 4, 4) FP32
     aiter_tensor_t* layer_input,   // (T, H) BF16 norm
     aiter_tensor_t* quant_out,     // (T, H) FP8 e4m3, mxfp8 only
     aiter_tensor_t* quant_scale,   // E8M0 bytes, mxfp8 only: (T, H/32) or M32K4 (pad32(T), H/32)
     aiter_tensor_t* next_pre,      // (T, 4) FP32
     aiter_tensor_t* residual,      // (T, 4, H) BF16
     aiter_tensor_t* fn_packed,     // mega_mhc_shuffle_fn(fn)
     aiter_tensor_t* hc_scale,      // (3,) FP32
     aiter_tensor_t* hc_base,       // (24,) FP32
     aiter_tensor_t* pre_mix,       // (T, 4) FP32, null when identity_pre
     aiter_tensor_t* sublayer_out,  // (T, H) BF16, null when has_post = 0
     aiter_tensor_t* post_mix,      // (T, 4) FP32, null when has_post = 0
     aiter_tensor_t* comb_mix,      // (T, 4, 4) FP32, null when has_post = 0
     aiter_tensor_t* norm_weight,   // (H,) BF16
     aiter_tensor_t* partials,      // FP32 workspace, see the table above
     float rms_eps,
     float hc_pre_eps,
     float hc_sinkhorn_eps,
     float hc_post_mult,
     float norm_eps,
     int64_t sinkhorn_repeat,
     int64_t tile_tokens,
     int mxfp8,
     int scale_row_major,
     int has_post,
     int identity_pre,
     int64_t cs,
     hipStream_t stream),
    (residual_out,
     post_out,
     comb_out,
     layer_input,
     quant_out,
     quant_scale,
     next_pre,
     residual,
     fn_packed,
     hc_scale,
     hc_base,
     pre_mix,
     sublayer_out,
     post_mix,
     comb_mix,
     norm_weight,
     partials,
     rms_eps,
     hc_pre_eps,
     hc_sinkhorn_eps,
     hc_post_mult,
     norm_eps,
     sinkhorn_repeat,
     tile_tokens,
     mxfp8,
     scale_row_major,
     has_post,
     identity_pre,
     cs,
     stream))
{
    AITER_CHECK(residual != nullptr && residual->is_gpu() && residual->dim() == 3 &&
                    residual->size(1) == 4,
                "residual must be a GPU tensor of shape (T, 4, H)");
    const int64_t t = residual->size(0);
    const int64_t h = residual->size(2);
    AITER_CHECK(h == MHC_HIDDEN, "mega_mhc asm supports H = 5120 only, got ", h);
    AITER_CHECK(
        t >= 0 && t <= std::numeric_limits<int32_t>::max() / 32, "unsupported token count ", t);
    AITER_CHECK(rms_eps > 0 && norm_eps > 0 && hc_sinkhorn_eps > 0 && sinkhorn_repeat > 0 &&
                    sinkhorn_repeat <= std::numeric_limits<int32_t>::max(),
                "RMS/Sinkhorn eps and sinkhorn_repeat must be positive");

    const mega_mhcConfig* cfg = find_config(
        tile_tokens, mxfp8 != 0, scale_row_major != 0, has_post != 0, identity_pre != 0, cs);
    AITER_CHECK(cfg != nullptr,
                "no mega_mhc asm kernel for tile_tokens=",
                tile_tokens,
                ", mxfp8=",
                mxfp8,
                ", scale_row_major=",
                scale_row_major,
                ", has_post=",
                has_post,
                ", identity_pre=",
                identity_pre,
                ", cs=",
                cs,
                " on ",
                get_gpu_arch());

    check_tensor(residual, residual, AITER_DTYPE_bf16, t * 4 * h, "residual");
    check_tensor(residual_out, residual, AITER_DTYPE_bf16, t * 4 * h, "residual_out");
    check_tensor(layer_input, residual, AITER_DTYPE_bf16, t * h, "layer_input");
    check_tensor(fn_packed, residual, AITER_DTYPE_i32, 24 * 4 * h, "fn_packed");
    check_tensor(hc_scale, residual, AITER_DTYPE_fp32, 3, "hc_scale");
    check_tensor(hc_base, residual, AITER_DTYPE_fp32, 24, "hc_base");
    check_tensor(norm_weight, residual, AITER_DTYPE_bf16, h, "norm_weight");
    check_tensor(next_pre, residual, AITER_DTYPE_fp32, t * 4, "next_pre");
    check_tensor(post_out, residual, AITER_DTYPE_fp32, t * 4, "post_out");
    check_tensor(comb_out, residual, AITER_DTYPE_fp32, t * 16, "comb_out");
    if(!identity_pre)
        check_tensor(pre_mix, residual, AITER_DTYPE_fp32, t * 4, "pre_mix");
    if(mxfp8)
    {
        check_tensor(quant_out, residual, AITER_DTYPE_fp8, t * h, "quant_out");
        check_tensor(quant_scale,
                     residual,
                     AITER_DTYPE_u8,
                     (scale_row_major ? t : (t + 31) / 32 * 32) * (h / 32),
                     "quant_scale");
    }
    if(has_post)
    {
        check_tensor(sublayer_out, residual, AITER_DTYPE_bf16, t * h, "sublayer_out");
        check_tensor(post_mix, residual, AITER_DTYPE_fp32, t * 4, "post_mix");
        check_tensor(comb_mix, residual, AITER_DTYPE_fp32, t * 16, "comb_mix");
        AITER_CHECK(t == 0 || residual_out->data_ptr() != residual->data_ptr(),
                    "residual_out must not alias residual");
    }
    else
    {
        AITER_CHECK(residual_out->data_ptr() == residual->data_ptr(),
                    "no_post returns the original residual");
    }
    const size_t partial_words = static_cast<size_t>(MHC_SPLITS) * t * MHC_PARTIAL_WORDS +
                                 (cfg->sqc ? static_cast<size_t>(MHC_SPLITS) * t : 0);
    check_tensor(partials, residual, AITER_DTYPE_fp32, partial_words, "partials");
    if(t == 0)
        return;

    const HipDeviceGuard device_guard(residual->device_id);
    void* dummy = hc_base->data_ptr(); // unread slots, as in the HIP BF16 launcher

    MegaMhcKernelArgs args;
    std::memset(&args, 0, sizeof(args));
    args.residual        = residual->data_ptr();
    args.sublayer        = has_post ? sublayer_out->data_ptr() : residual->data_ptr();
    args.post_mix        = has_post ? post_mix->data_ptr() : dummy;
    args.comb_mix        = has_post ? comb_mix->data_ptr() : dummy;
    args.pre_mix         = identity_pre ? dummy : pre_mix->data_ptr();
    args.fn_packed       = fn_packed->data_ptr();
    args.tokens          = static_cast<int32_t>(t);
    args.split_arg       = MHC_SPLITS;
    args.residual_out    = residual_out->data_ptr();
    args.post_out        = post_out->data_ptr();
    args.comb_out        = comb_out->data_ptr();
    args.layer_input     = layer_input->data_ptr();
    args.next_pre        = next_pre->data_ptr();
    args.hc_scale        = hc_scale->data_ptr();
    args.hc_base         = hc_base->data_ptr();
    args.norm_weight     = norm_weight->data_ptr();
    args.partials        = partials->data_ptr();
    args.counters        = dummy;
    args.hidden_arg      = static_cast<int32_t>(h);
    args.rms_eps         = rms_eps;
    args.pre_eps         = hc_pre_eps;
    args.sinkhorn_eps    = hc_sinkhorn_eps;
    args.post_mult       = hc_post_mult;
    args.norm_eps        = norm_eps;
    args.sinkhorn_repeat = static_cast<int32_t>(sinkhorn_repeat);
    args.tail_token0     = static_cast<int32_t>(t);
    args.quant_out       = mxfp8 ? quant_out->data_ptr() : dummy;
    args.quant_scale     = mxfp8 ? quant_scale->data_ptr() : dummy;
    args.scale_row_major = mxfp8 && scale_row_major ? 1 : 0;
    size_t arg_size      = sizeof(args);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(cfg->knl_name, [&]() {
        return AiterAsmKernel(cfg->knl_name.c_str(), cfg->co_name.c_str());
    });

    // A persistent kernel steps by ps_clusters tiles per round, so only the clusters
    // that own a tile in the first round need to launch.
    const int64_t tiles    = (t + cfg->tile_tokens - 1) / cfg->tile_tokens;
    const int64_t clusters = cfg->persistent ? std::min<int64_t>(cfg->ps_clusters, tiles) : tiles;
    const int gdx          = static_cast<int>(clusters * MHC_SPLITS);
    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, 1, 1, MHC_BLOCK, 1, 1, stream, MHC_SPLITS, 1, 1});
}
