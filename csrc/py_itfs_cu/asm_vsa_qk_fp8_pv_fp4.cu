// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP launcher for the QK=FP8 / PV=FP4 mixed-precision VSA block-sparse
// attention kernel.  Uses a local AsmKernel variant that passes the kernel's
// dynamic LDS size (the canonical `AiterAsmKernel` in aiter_hip_common.h
// hard-codes sharedMemBytes = 0, which would crash this kernel — it needs
// 36 KB of dynamic LDS for the Q/K/V double-buffered tiles).

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "vsa_qk_fp8_pv_fp4.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

namespace {

// ---------------------------------------------------------------------------
// Kernel constants — must match what the .co was compiled with
// (see /home/vsa_qk_fp8_pv_fp4_hip/vsa_qk_fp8_pv_fp4.hip).
// ---------------------------------------------------------------------------
constexpr int kBlockX  = 256;       // 4 wavefronts
constexpr int kLdsBytes = 36864;    // dynamic LDS bytes per workgroup
constexpr int kNumCus  = 256;       // MI355X CU count
constexpr int kOcc     = 2;         // waves/EU target -> 2 workgroups/CU
constexpr int kGridCap = kNumCus * kOcc;   // 512 outer waves

// The .co is built from a single templated implementation with three
// NUM_HEADS_T instantiations exposed as separate __global__ shims:
//     vsa_qk_fp8_pv_fp4_kernel_h1   (NUM_HEADS=1)
//     vsa_qk_fp8_pv_fp4_kernel_h2   (NUM_HEADS=2)
//     vsa_qk_fp8_pv_fp4_kernel_h4   (NUM_HEADS=4)
// NUM_HEADS appears in exactly one place inside the kernel:
//     total_tiles = B * NUM_HEADS * num_q_blks
// so the three variants share IDENTICAL SASS / VGPR / LDS footprint;
// the only difference is the immediate operand of one IMUL.
// The launcher picks the variant whose NUM_HEADS is the largest divisor
// of BH in {1, 2, 4}, and forwards B_eff = BH / NUM_HEADS, so
// total_tiles = B_eff * NUM_HEADS * num_q_blks == BH * num_q_blks
// for arbitrary H without changing per-tile work.

// ---------------------------------------------------------------------------
// 400-byte kernarg layout — must match the kernel signature
//   __global__ void vsa_qk_fp8_pv_fp4_kernel(
//       int B, int T, int num_q_blks, int max_kv_blks,
//       const int32_t* logical_idx_mapping,   // lim
//       const uint8_t* Q,
//       const uint8_t* K,
//       const uint8_t* V,
//       const int32_t* q2k_index,
//       const int32_t* q2k_num,
//       const int32_t* variable_block_sizes,  // vbs
//       const uint8_t* qscale,
//       const uint8_t* kscale,
//       const uint8_t* vscale,
//       hip_bfloat16*  Out,
//       float*         Lse,
//       int32_t*       d_counter,
//       int32_t*       s_counter,
//       int            n_dense)
// ---------------------------------------------------------------------------
struct __attribute__((packed)) KernelArgs {
    int32_t B;
    int32_t T;
    int32_t num_q_blks;
    int32_t max_kv_blks;
    void*   logical_idx_mapping;
    void*   Q;
    void*   K;
    void*   V;
    void*   q2k_index;
    void*   q2k_num;
    void*   variable_block_sizes;
    void*   qscale;
    void*   kscale;
    void*   vscale;
    void*   Out;
    void*   Lse;
    void*   d_counter;
    void*   s_counter;
    int32_t n_dense;
    char    _pad[400 - (4 * 4 + 8 * 14 + 4)];
};
static_assert(sizeof(KernelArgs) == 400,
              "VSA QK-FP8 PV-FP4 KernelArgs must be 400 bytes");

// ---------------------------------------------------------------------------
// AsmKernel variant that passes a non-zero dynamic LDS size to
// hipModuleLaunchKernel (the canonical `AiterAsmKernel` in
// aiter_hip_common.h hard-codes sharedMemBytes = 0).
// ---------------------------------------------------------------------------
class VsaQkFp8PvFp4AsmKernel {
   public:
    // Loads ONE .co containing all three NUM_HEADS_T variants
    // (`<base>_h1`, `<base>_h2`, `<base>_h4`).  The selected entry-point
    // is then picked per call by `func_for_bh(BH)`.
    VsaQkFp8PvFp4AsmKernel(const char* base_name, const char* hsaco) {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        AITER_CHECK(AITER_ASM_DIR != nullptr,
                    "AITER_ASM_DIR not set (needed to locate ", hsaco, ")");
        const std::string full_path =
            std::string(AITER_ASM_DIR) + "/gfx950/" + hsaco;
        std::cout << "[aiter] hipModuleLoad: " << full_path
                  << " GetFunction: " << base_name << "_h{1,2,4}";
        HIP_CALL(hipModuleLoad(&module_, full_path.c_str()));

        const std::string b = base_name;
        HIP_CALL(hipModuleGetFunction(&func_h1_, module_, (b + "_h1").c_str()));
        HIP_CALL(hipModuleGetFunction(&func_h2_, module_, (b + "_h2").c_str()));
        HIP_CALL(hipModuleGetFunction(&func_h4_, module_, (b + "_h4").c_str()));
        std::cout << " Success" << std::endl;
    }

    ~VsaQkFp8PvFp4AsmKernel() { HIP_CALL(hipModuleUnload(module_)); }

    // Pick the variant whose NUM_HEADS is the largest divisor of BH
    // in {1, 2, 4}, and return both the function handle and the
    // corresponding NUM_HEADS so the caller can derive B_eff.
    struct Selected { hipFunction_t func; int num_heads; };
    Selected select_for_bh(int64_t BH) const {
        if (BH % 4 == 0) return {func_h4_, 4};
        if (BH % 2 == 0) return {func_h2_, 2};
        return {func_h1_, 1};
    }

    static void launch_one(hipFunction_t   func,
                           void*           args,
                           size_t          arg_size,
                           int             gdx,
                           int             bdx,
                           int             lds_bytes,
                           hipStream_t     stream) {
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size,
                          HIP_LAUNCH_PARAM_END};
        HIP_CALL(hipModuleLaunchKernel(func,
                                       gdx, 1, 1,
                                       bdx, 1, 1,
                                       lds_bytes,    // dynamic LDS
                                       stream,
                                       nullptr,
                                       config));
    }

   private:
    hipModule_t   module_  = nullptr;
    hipFunction_t func_h1_ = nullptr;
    hipFunction_t func_h2_ = nullptr;
    hipFunction_t func_h4_ = nullptr;
};

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void vsa_qk_fp8_pv_fp4(const torch::Tensor& q,
                       const torch::Tensor& k,
                       const torch::Tensor& v,
                       const torch::Tensor& qscale,
                       const torch::Tensor& kscale,
                       const torch::Tensor& vscale,
                       const torch::Tensor& q2k_idx,
                       const torch::Tensor& q2k_num,
                       const torch::Tensor& vbs,
                       const torch::Tensor& lim,
                       const torch::Tensor& out,
                       const torch::Tensor& lse,
                       const torch::Tensor& counters,
                       int64_t B,
                       int64_t T,
                       int64_t num_q_blks,
                       int64_t max_kv,
                       int64_t n_dense) {
    TORCH_CHECK(counters.dtype() == torch::kInt32 && counters.numel() >= 2,
                "vsa_qk_fp8_pv_fp4: counters must be int32 with >= 2 elements");
    TORCH_CHECK(counters.is_contiguous(),
                "vsa_qk_fp8_pv_fp4: counters must be contiguous");

    const int64_t BH = q.size(0);
    TORCH_CHECK(BH % B == 0,
                "vsa_qk_fp8_pv_fp4: q.size(0)=", BH,
                " must be divisible by B=", B);
    const int64_t total_tiles = BH * num_q_blks;
    TORCH_CHECK(total_tiles > 0,
                "vsa_qk_fp8_pv_fp4: empty workload (BH*num_q_blks == 0)");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(q));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Zero the dispatch counters before each launch (kernel relies on this).
    HIP_CALL(hipMemsetAsync(counters.data_ptr(), 0,
                            counters.numel() * sizeof(int32_t),
                            stream));

    // Pick the kernel variant that exactly divides BH (see Selected{}
    // doc in VsaQkFp8PvFp4AsmKernel above).  B_eff = BH / NUM_HEADS so
    // the kernel's `total_tiles = B * NUM_HEADS * num_q_blks` evaluates
    // to BH * num_q_blks regardless of the user's real (B, H) split.
    static VsaQkFp8PvFp4AsmKernel impl("vsa_qk_fp8_pv_fp4_kernel",
                                       "vsa/vsa_qk_fp8_pv_fp4.co");
    const auto picked    = impl.select_for_bh(BH);
    const int32_t B_eff  = static_cast<int32_t>(BH / picked.num_heads);

    KernelArgs a{};
    a.B                    = B_eff;
    a.T                    = static_cast<int32_t>(T);
    a.num_q_blks           = static_cast<int32_t>(num_q_blks);
    a.max_kv_blks          = static_cast<int32_t>(max_kv);
    a.logical_idx_mapping  = lim.data_ptr();
    a.Q                    = q.data_ptr();
    a.K                    = k.data_ptr();
    a.V                    = v.data_ptr();
    a.q2k_index            = q2k_idx.data_ptr();
    a.q2k_num              = q2k_num.data_ptr();
    a.variable_block_sizes = vbs.data_ptr();
    a.qscale               = qscale.data_ptr();
    a.kscale               = kscale.data_ptr();
    a.vscale               = vscale.data_ptr();
    a.Out                  = out.data_ptr();
    a.Lse                  = lse.data_ptr();
    a.d_counter            = counters.data_ptr();
    a.s_counter            = reinterpret_cast<int32_t*>(counters.data_ptr()) + 1;
    a.n_dense              = static_cast<int32_t>(n_dense);

    const int grid_x = (total_tiles < kGridCap)
                           ? static_cast<int>(total_tiles)
                           : kGridCap;
    VsaQkFp8PvFp4AsmKernel::launch_one(
        picked.func, &a, sizeof(a), grid_x, kBlockX, kLdsBytes, stream);
}
