// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 delayed mHC seam kernels (hsa/gfx942/mhc_seam/) for
// DeepSeek-V4.1-Flash: hc_mult = 4, hidden_size = 5120.
//
// For T tokens, with x the sub-layer output and post / comb / pre the gates of the
// previous seam:
//   residual_out[t, j] = bf16(sum_i comb[t, i, j] residual[t, i] + post[t, j] x[t])
//                        fp32, the order and truncation of aiter's mhc_post (bit-exact)
//   layer_input[t]     = bf16(((0 + pre[t, 0] r0) + pre[t, 1] r1) + ... + pre[t, 3] r3)
//                        r_j = fp32(residual_out[t, j]), separate fp32 mul / add, RNE
// and the split-K partials of the next pre projection over column slices of width w:
//   part[c, t, 0:24]  = residual_out[t, :, w c : w c + w] . fn[:, same columns]^T
//   part[c, t, 24]    = sum of residual_out[t, :, w c : w c + w]^2
//   part[c, t, 25:32] = 0
// with w = 512 (fused kernel, persistent grid, part (10, T, 32)) or w = 128 (few-token
// kernel, one wave per slice and token, part (40, T, 32)). The gates kernel turns either
// into the next post / comb / pre.
//
// The kernels take raw pointers and T, and build their 128-bit buffer resource
// descriptors themselves: each one is rebased to the workgroup's first token with
// num_records clipped to its valid tokens, so rows past T are never read or written.
// Offsets are 32 bits, so one launch covers < 4 GiB per tensor.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mhc_seam_configs.hpp"
#include <cstring>
#include <memory>

namespace {

constexpr int kHcMult    = 4;
constexpr int kHidden    = 5120;
constexpr int kChunk     = 512; // h per wave of the fused kernel
constexpr int kNumChunks = kHidden / kChunk;
constexpr int kMixes     = 2 * kHcMult + kHcMult * kHcMult; // 24
constexpr int kPartCols  = 32;
constexpr int kSlice     = 128; // h per wave of the few-token kernel
constexpr int kNumSlices = kHidden / kSlice;
constexpr int kGatesTokPerWg = 8;

// Kernel arguments as the code objects read them: one 16-byte slot per argument
// (pointer + p2, uint32 + p3), pointers first; the sizes are the code objects'
// kernarg_segment_size.
struct __attribute__((packed)) SeamFusedArgs
{
    const void* ptr_R;
    p2 _p0;
    const void* ptr_X;
    p2 _p1;
    void* ptr_O;
    p2 _p2;
    void* ptr_L;
    p2 _p3;
    const void* ptr_C;
    p2 _p4;
    const void* ptr_P;
    p2 _p5;
    const void* ptr_Q;
    p2 _p6;
    const void* ptr_F; // fn (24, 4 H) fp32
    p2 _p7;
    void* ptr_S; // part (10, T, 32) fp32
    p2 _p8;
    unsigned int dim_T;
    p3 _p9;
    unsigned int tpw; // tokens per workgroup, a multiple of 4
    p3 _p10;
};
static_assert(sizeof(SeamFusedArgs) == 176, "mhc_seam fused kernarg size");

struct __attribute__((packed)) SeamGatesArgs
{
    const void* ptr_S; // part (S, T, 32) fp32
    p2 _p0;
    const void* ptr_A; // hc_scale (3,) fp32
    p2 _p1;
    const void* ptr_B; // hc_base (24,) fp32
    p2 _p2;
    void* ptr_PO; // post_mix (T, 4) fp32
    p2 _p3;
    void* ptr_CO; // comb_mix (T, 4, 4) fp32
    p2 _p4;
    void* ptr_PR; // pre_mix (T, 4) fp32
    p2 _p5;
    unsigned int dim_T;
    p3 _p6;
    unsigned int dim_S;
    p3 _p7;
    unsigned int repeat; // sinkhorn_repeat
    p3 _p8;
    float rms_eps;
    p3 _p9;
    float pre_eps;
    p3 _p10;
    float sinkhorn_eps;
    p3 _p11;
    float post_mult;
    p3 _p12;
    float inv_k; // 1 / (hc_mult * hidden_size)
    p3 _p13;
};
static_assert(sizeof(SeamGatesArgs) == 224, "mhc_seam gates kernarg size");

// The kernels address a tensor with 32-bit offsets from a 48-bit base.
void* checked_ptr(const aiter_tensor_t* t, const char* name)
{
    const uint64_t nbytes = static_cast<uint64_t>(t->numel()) * t->element_size();
    AITER_CHECK(nbytes < (1ull << 32),
                "mhc_seam: ",
                name,
                " spans ",
                nbytes,
                " bytes; a launch covers less than 4 GiB per tensor");
    const uint64_t addr = reinterpret_cast<uint64_t>(t->data_ptr());
    AITER_CHECK(addr < (1ull << 48), "mhc_seam: ", name, " address exceeds 48 bits");
    return t->data_ptr();
}

void check_tensor(const aiter_tensor_t* t,
                  const char* name,
                  AiterDtype dtype,
                  std::initializer_list<int64_t> shape)
{
    AITER_CHECK(t != nullptr, "mhc_seam: ", name, " is required");
    AITER_CHECK(t->dtype() == dtype,
                "mhc_seam: ",
                name,
                " has dtype ",
                AiterDtype_to_str(t->dtype()),
                ", expected ",
                AiterDtype_to_str(dtype));
    AITER_CHECK(t->dim() == static_cast<int>(shape.size()), "mhc_seam: ", name, " rank");
    int d = 0;
    for(int64_t s : shape)
    {
        AITER_CHECK(
            t->size(d) == s, "mhc_seam: ", name, " dim ", d, " is ", t->size(d), ", expected ", s);
        ++d;
    }
    AITER_CHECK(t->is_contiguous(), "mhc_seam: ", name, " must be contiguous");
}

// The seam inputs shared by the seam kernels; returns T.
int64_t check_seam_inputs(const aiter_tensor_t* residual_out,
                          const aiter_tensor_t* layer_input,
                          const aiter_tensor_t* residual,
                          const aiter_tensor_t* sublayer_out,
                          const aiter_tensor_t* post_layer_mix,
                          const aiter_tensor_t* comb_res_mix,
                          const aiter_tensor_t* pre_mix)
{
    AITER_CHECK(residual != nullptr && residual->dim() == 3, "mhc_seam: residual is (T, 4, H)");
    const int64_t T = residual->size(0);
    check_tensor(residual, "residual", AITER_DTYPE_bf16, {T, kHcMult, kHidden});
    check_tensor(residual_out, "residual_out", AITER_DTYPE_bf16, {T, kHcMult, kHidden});
    check_tensor(sublayer_out, "sublayer_out", AITER_DTYPE_bf16, {T, kHidden});
    check_tensor(layer_input, "layer_input", AITER_DTYPE_bf16, {T, kHidden});
    check_tensor(post_layer_mix, "post_layer_mix", AITER_DTYPE_fp32, {T, kHcMult});
    check_tensor(comb_res_mix, "comb_res_mix", AITER_DTYPE_fp32, {T, kHcMult, kHcMult});
    check_tensor(pre_mix, "pre_mix", AITER_DTYPE_fp32, {T, kHcMult});
    AITER_CHECK(residual_out->data_ptr() != residual->data_ptr(),
                "mhc_seam: residual_out must not alias residual");
    return T;
}

const mhc_seamConfig&
get_kernel_cfg(const std::string& arch_id, const std::string& kind, int store_nt)
{
    const CFG* cfgs = &cfg_mhc_seam;
    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.kind == kind && cfg.store_nt == store_nt)
            return cfg;
    }
    AITER_CHECK(false,
                "mhc_seam: no ",
                kind,
                " kernel with store_nt=",
                store_nt,
                " for ",
                arch_id,
                " (the seam kernels are gfx942 only)");
    return cfgs->begin()->second; // unreachable
}

AiterAsmKernel* get_kernel(const mhc_seamConfig& cfg)
{
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    return &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
}

} // namespace

// Entry points called through ctypes: a failed AITER_CHECK is caught at the C boundary
// and raised in Python as RuntimeError.
AITER_CTYPES_ERROR_DEF

// residual_out (T, 4, 5120) bf16, layer_input (T, 5120) bf16 and the split-K partials of
// the next pre projection <- residual (T, 4, 5120) bf16, sublayer_out (T, 5120) bf16,
// post_layer_mix (T, 4) fp32, comb_res_mix (T, 4, 4) fp32, pre_mix (T, 4) fp32 and
// fn (24, 4 * 5120) fp32; part (10, T, 32) fp32. Persistent grid: 10 chunks x
// `workers` token ranges (workers <= 0: CUs / 10), one 4-wave workgroup per CU.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(mhc_seam_fused_asm,
                                    (aiter_tensor_t* residual_out,
                                     aiter_tensor_t* layer_input,
                                     aiter_tensor_t* part,
                                     aiter_tensor_t* residual,
                                     aiter_tensor_t* sublayer_out,
                                     aiter_tensor_t* post_layer_mix,
                                     aiter_tensor_t* comb_res_mix,
                                     aiter_tensor_t* pre_mix,
                                     aiter_tensor_t* fn,
                                     int64_t workers,
                                     int store_nt,
                                     hipStream_t stream),
                                    (residual_out,
                                     layer_input,
                                     part,
                                     residual,
                                     sublayer_out,
                                     post_layer_mix,
                                     comb_res_mix,
                                     pre_mix,
                                     fn,
                                     workers,
                                     store_nt,
                                     stream))
{
    const int64_t T = check_seam_inputs(
        residual_out, layer_input, residual, sublayer_out, post_layer_mix, comb_res_mix, pre_mix);
    check_tensor(part, "part", AITER_DTYPE_fp32, {kNumChunks, T, kPartCols});
    check_tensor(fn, "fn", AITER_DTYPE_fp32, {kMixes, kHcMult * kHidden});
    if(T == 0)
        return;
    const HipDeviceGuard device_guard(residual->device_id);
    const auto& cfg      = get_kernel_cfg(get_gpu_arch(), "fused", store_nt ? 1 : 0);
    AiterAsmKernel* impl = get_kernel(cfg);

    if(workers <= 0)
        workers = std::max<int64_t>(1, get_num_cu_func() / kNumChunks);
    int64_t tpw       = (T + workers - 1) / workers;
    tpw               = (tpw + 3) / 4 * 4; // whole 4-token tiles
    const int64_t gdy = (T + tpw - 1) / tpw;

    SeamFusedArgs args;
    std::memset(&args, 0, sizeof(args));
    size_t arg_size = sizeof(args);
    args.ptr_R      = checked_ptr(residual, "residual");
    args.ptr_X      = checked_ptr(sublayer_out, "sublayer_out");
    args.ptr_O      = checked_ptr(residual_out, "residual_out");
    args.ptr_L      = checked_ptr(layer_input, "layer_input");
    args.ptr_C      = checked_ptr(comb_res_mix, "comb_res_mix");
    args.ptr_P      = checked_ptr(post_layer_mix, "post_layer_mix");
    args.ptr_Q      = checked_ptr(pre_mix, "pre_mix");
    args.ptr_F      = checked_ptr(fn, "fn");
    args.ptr_S      = checked_ptr(part, "part");
    args.dim_T      = static_cast<unsigned int>(T);
    args.tpw        = static_cast<unsigned int>(tpw);

    impl->launch_kernel({&args,
                         &arg_size,
                         kNumChunks,            // gdx: 512-h chunks
                         static_cast<int>(gdy), // gdy: token ranges
                         1,                     // gdz
                         cfg.block,             // bdx: 4 x wave64
                         1,                     // bdy
                         1,                     // bdz
                         stream});
}

// Few-token variant of mhc_seam_fused_asm: the same outputs with the partials over 128-h
// slices, part (40, T, 32) fp32. One wave per (slice, token), no persistent grid.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(mhc_seam_small_asm,
                                    (aiter_tensor_t* residual_out,
                                     aiter_tensor_t* layer_input,
                                     aiter_tensor_t* part,
                                     aiter_tensor_t* residual,
                                     aiter_tensor_t* sublayer_out,
                                     aiter_tensor_t* post_layer_mix,
                                     aiter_tensor_t* comb_res_mix,
                                     aiter_tensor_t* pre_mix,
                                     aiter_tensor_t* fn,
                                     hipStream_t stream),
                                    (residual_out,
                                     layer_input,
                                     part,
                                     residual,
                                     sublayer_out,
                                     post_layer_mix,
                                     comb_res_mix,
                                     pre_mix,
                                     fn,
                                     stream))
{
    const int64_t T = check_seam_inputs(
        residual_out, layer_input, residual, sublayer_out, post_layer_mix, comb_res_mix, pre_mix);
    check_tensor(part, "part", AITER_DTYPE_fp32, {kNumSlices, T, kPartCols});
    check_tensor(fn, "fn", AITER_DTYPE_fp32, {kMixes, kHcMult * kHidden});
    if(T == 0)
        return;
    AITER_CHECK(T < 65536, "mhc_seam_small_asm: T too large for one launch: ", T);
    const HipDeviceGuard device_guard(residual->device_id);
    const auto& cfg      = get_kernel_cfg(get_gpu_arch(), "small", 0);
    AiterAsmKernel* impl = get_kernel(cfg);

    SeamFusedArgs args;
    std::memset(&args, 0, sizeof(args));
    size_t arg_size = sizeof(args);
    args.ptr_R      = checked_ptr(residual, "residual");
    args.ptr_X      = checked_ptr(sublayer_out, "sublayer_out");
    args.ptr_O      = checked_ptr(residual_out, "residual_out");
    args.ptr_L      = checked_ptr(layer_input, "layer_input");
    args.ptr_C      = checked_ptr(comb_res_mix, "comb_res_mix");
    args.ptr_P      = checked_ptr(post_layer_mix, "post_layer_mix");
    args.ptr_Q      = checked_ptr(pre_mix, "pre_mix");
    args.ptr_F      = checked_ptr(fn, "fn");
    args.ptr_S      = checked_ptr(part, "part");
    args.dim_T      = static_cast<unsigned int>(T);
    args.tpw        = 1;

    impl->launch_kernel({&args,
                         &arg_size,
                         kNumSlices,          // gdx: 128-h slices
                         static_cast<int>(T), // gdy: tokens
                         1,                   // gdz
                         cfg.block,           // bdx: one wave
                         1,                   // bdy
                         1,                   // bdz
                         stream});
}

// The gates of the next pre projection from split-K partials part (S, T, 32) fp32
// (columns 0..23 the mixes, column 24 the sum of squares; either kernel's part):
//   rstd     = rsqrt(sum_s part[s, t, 24] / hc_hidden_size + rms_eps)
//   pre_mix  = sigmoid(mixes[0:4] rstd hc_scale[0] + hc_base[0:4]) + hc_pre_eps
//   post_mix = sigmoid(mixes[4:8] rstd hc_scale[1] + hc_base[4:8]) * hc_post_mult_value
//   comb_mix = Sinkhorn(mixes[8:24] rstd hc_scale[2] + hc_base[8:24], sinkhorn_repeat)
// as mhc_pre_big_fuse computes them. Two tokens per wave.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(mhc_seam_gates_asm,
                                    (aiter_tensor_t* post_mix,
                                     aiter_tensor_t* comb_mix,
                                     aiter_tensor_t* pre_mix,
                                     aiter_tensor_t* part,
                                     aiter_tensor_t* hc_scale,
                                     aiter_tensor_t* hc_base,
                                     float rms_eps,
                                     float hc_pre_eps,
                                     float hc_sinkhorn_eps,
                                     float hc_post_mult_value,
                                     int64_t sinkhorn_repeat,
                                     int64_t hc_hidden_size,
                                     hipStream_t stream),
                                    (post_mix,
                                     comb_mix,
                                     pre_mix,
                                     part,
                                     hc_scale,
                                     hc_base,
                                     rms_eps,
                                     hc_pre_eps,
                                     hc_sinkhorn_eps,
                                     hc_post_mult_value,
                                     sinkhorn_repeat,
                                     hc_hidden_size,
                                     stream))
{
    AITER_CHECK(part != nullptr && part->dim() == 3, "mhc_seam: part is (S, T, 32)");
    const int64_t S = part->size(0);
    const int64_t T = part->size(1);
    check_tensor(part, "part", AITER_DTYPE_fp32, {S, T, kPartCols});
    check_tensor(post_mix, "post_mix", AITER_DTYPE_fp32, {T, kHcMult, 1});
    check_tensor(comb_mix, "comb_mix", AITER_DTYPE_fp32, {T, kHcMult, kHcMult});
    check_tensor(pre_mix, "pre_mix", AITER_DTYPE_fp32, {T, kHcMult});
    check_tensor(hc_scale, "hc_scale", AITER_DTYPE_fp32, {3});
    check_tensor(hc_base, "hc_base", AITER_DTYPE_fp32, {kMixes});
    AITER_CHECK(S >= 1 && S <= 64, "mhc_seam_gates_asm: 1 to 64 splits, got ", S);
    AITER_CHECK(sinkhorn_repeat >= 1, "mhc_seam_gates_asm: sinkhorn_repeat must be >= 1");
    AITER_CHECK(hc_hidden_size > 0, "mhc_seam_gates_asm: hc_hidden_size must be > 0");
    if(T == 0)
        return;
    const HipDeviceGuard device_guard(part->device_id);
    const auto& cfg      = get_kernel_cfg(get_gpu_arch(), "gates", 0);
    AiterAsmKernel* impl = get_kernel(cfg);

    SeamGatesArgs args;
    std::memset(&args, 0, sizeof(args));
    size_t arg_size   = sizeof(args);
    args.ptr_S        = checked_ptr(part, "part");
    args.ptr_A        = checked_ptr(hc_scale, "hc_scale");
    args.ptr_B        = checked_ptr(hc_base, "hc_base");
    args.ptr_PO       = checked_ptr(post_mix, "post_mix");
    args.ptr_CO       = checked_ptr(comb_mix, "comb_mix");
    args.ptr_PR       = checked_ptr(pre_mix, "pre_mix");
    args.dim_T        = static_cast<unsigned int>(T);
    args.dim_S        = static_cast<unsigned int>(S);
    args.repeat       = static_cast<unsigned int>(sinkhorn_repeat);
    args.rms_eps      = rms_eps;
    args.pre_eps      = hc_pre_eps;
    args.sinkhorn_eps = hc_sinkhorn_eps;
    args.post_mult    = hc_post_mult_value;
    args.inv_k        = 1.0f / static_cast<float>(hc_hidden_size);

    const int64_t gdx = (T + kGatesTokPerWg - 1) / kGatesTokPerWg;
    impl->launch_kernel({&args,
                         &arg_size,
                         static_cast<int>(gdx), // gdx: 8 tokens per workgroup
                         1,                     // gdy
                         1,                     // gdz
                         cfg.block,             // bdx: 4 x wave64
                         1,                     // bdy
                         1,                     // bdz
                         stream});
}
