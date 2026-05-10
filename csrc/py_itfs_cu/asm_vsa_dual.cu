// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Launcher for the FP4 VSA dual-warp-set attention kernel
// (loads /home/aiter/hsa/gfx950/vsa/vsa_dual_setprio_dropB.co).
//
// Mirrors the topk_per_row_decode_fast pattern (AiterAsmKernel registers
// the .co once per process via __hipRegisterFatBinary; per-call work is a
// 400-byte stack-local kernarg + hipModuleLaunchKernel), but uses a local
// dynamic-LDS-aware launch helper because the .co's
// group_segment_fixed_size = 0 — the 27152-byte LDS must be passed via
// `sharedMemBytes` at launch time and AiterAsmKernel hard-codes that to 0.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "vsa_dual.h"

namespace {

// 400-byte HSA kernarg layout (must match the .co's expected ABI).
struct __attribute__((packed)) VsaDualKernArg
{
    int32_t B;
    int32_t T;
    int32_t num_q_blks;
    int32_t max_kv;
    void*   lim;
    void*   Q;
    void*   K;
    void*   V;
    void*   q2k_idx;
    void*   q2k_num;
    void*   vbs;
    void*   qscale;
    void*   kscale;
    void*   vmean;
    void*   vscale;
    void*   Out;
    void*   Lse;
    void*   d_counter;
    void*   s_counter;
    int32_t n_dense;
    char    _pad[400 - (4 * 4 + 8 * 15 + 4)];
};
static_assert(sizeof(VsaDualKernArg) == 400, "VsaDualKernArg must be 400 bytes");

// Local AiterAsmKernel-equivalent that supports dynamic LDS at launch.
// Identical fat-binary registration to the one in aiter_hip_common.h, but
// passes `sharedMemBytes` to hipModuleLaunchKernel.
class VsaDualAsmKernel
{
   public:
    VsaDualAsmKernel(const char* kernel_name, const char* hsaco_path)
    {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::string arch_name     = get_gpu_arch();
        AITER_CHECK(AITER_ASM_DIR != nullptr,
                    "AITER_ASM_DIR not set (needed to locate ", hsaco_path, ")");
        std::string full_path =
            std::string(AITER_ASM_DIR) + "/" + arch_name + "/" + hsaco_path;
        AITER_LOG_INFO("LoadKernel: " << kernel_name << " hsaco: " << full_path);

        std::ifstream file(full_path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "failed to open ", full_path.c_str());
        size_t file_size = file.tellg();
        hsaco_data_.reset(new char[file_size]);
        file.seekg(0, std::ios::beg);
        AITER_CHECK(file.read(hsaco_data_.get(), file_size),
                    "failed to read ", full_path.c_str());

        aiter_detail::FatBinaryWrapper fat_bin{};
        fat_bin.binary = hsaco_data_.get();
        module_        = aiter_detail::__hipRegisterFatBinary(&fat_bin);
        AITER_CHECK(module_ != nullptr,
                    "failed to load module for ", kernel_name);
        aiter_detail::__hipRegisterFunction(module_,
                                            static_cast<void*>(this),
                                            kernel_name,
                                            kernel_name,
                                            -1,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr);
    }

    ~VsaDualAsmKernel()
    {
        aiter_detail::__hipUnregisterFatBinary(module_);
    }

    VsaDualAsmKernel(const VsaDualAsmKernel&)            = delete;
    VsaDualAsmKernel& operator=(const VsaDualAsmKernel&) = delete;

    void launch(void* args_ptr, size_t* arg_size_ptr,
                int gdx, int gdy, int gdz,
                int bdx, int bdy, int bdz,
                int shared_mem_bytes,
                hipStream_t stream)
    {
        void* config[]            = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args_ptr,
                                     HIP_LAUNCH_PARAM_BUFFER_SIZE,    arg_size_ptr,
                                     HIP_LAUNCH_PARAM_END};
        hipFunction_t kernel_func = nullptr;
        (void)hipGetFuncBySymbol(&kernel_func, reinterpret_cast<void*>(this));

        HIP_CALL_LAUNCH(hipModuleLaunchKernel(kernel_func,
                                              gdx, gdy, gdz,
                                              bdx, bdy, bdz,
                                              shared_mem_bytes,
                                              stream,
                                              nullptr,
                                              (void**)&config));
    }

   private:
    void* module_ = nullptr;
    std::unique_ptr<char[]> hsaco_data_;
};

constexpr int kBlockX   = 256;
constexpr int kNumCUs   = 256;
constexpr int kOcc      = 2;
constexpr int kGridCap  = kNumCUs * kOcc;
constexpr int kLdsBytes = 27152;

}  // namespace

void vsa_dual_dropB(const torch::Tensor& q,
                    const torch::Tensor& k,
                    const torch::Tensor& v,
                    const torch::Tensor& qscale,
                    const torch::Tensor& kscale,
                    const torch::Tensor& vmean,
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
                    int64_t n_dense)
{
    TORCH_CHECK(counters.dtype() == torch::kInt32 && counters.numel() >= 2,
                "counters must be an int32 tensor with >= 2 elements");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(q.dim() >= 1, "q must be at least 1-D (BH, T, D)");

    // q is laid out as (BH, T, D); BH = B * H is implicit from q.shape[0].
    const int64_t BH = q.size(0);

    // .data_ptr() on a const Tensor is fine — the kernel mutates the
    // underlying storage of `out`, `lse`, and `counters` directly.
    void* d_counter_ptr = counters.data_ptr();
    void* s_counter_ptr = static_cast<char*>(d_counter_ptr) + sizeof(int32_t);

    VsaDualKernArg args;
    std::memset(&args, 0, sizeof(args));
    args.B          = static_cast<int32_t>(B);
    args.T          = static_cast<int32_t>(T);
    args.num_q_blks = static_cast<int32_t>(num_q_blks);
    args.max_kv     = static_cast<int32_t>(max_kv);
    args.lim        = lim.data_ptr();
    args.Q          = q.data_ptr();
    args.K          = k.data_ptr();
    args.V          = v.data_ptr();
    args.q2k_idx    = q2k_idx.data_ptr();
    args.q2k_num    = q2k_num.data_ptr();
    args.vbs        = vbs.data_ptr();
    args.qscale     = qscale.data_ptr();
    args.kscale     = kscale.data_ptr();
    args.vmean      = vmean.data_ptr();
    args.vscale     = vscale.data_ptr();
    args.Out        = out.data_ptr();
    args.Lse        = lse.data_ptr();
    args.d_counter  = d_counter_ptr;
    args.s_counter  = s_counter_ptr;
    args.n_dense    = static_cast<int32_t>(n_dense);

    size_t arg_size = sizeof(args);

    static VsaDualAsmKernel impl_vsa_dual(
        "fp4_vsa_dual_kernel",
        "/vsa/vsa_dual_setprio_dropB.co");

    const int total_tiles = static_cast<int>(BH) * static_cast<int>(num_q_blks);
    const int grid_x = total_tiles < kGridCap ? total_tiles : kGridCap;

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    const_cast<torch::Tensor&>(counters).zero_();

    impl_vsa_dual.launch(&args, &arg_size,
                         grid_x, 1, 1,
                         kBlockX, 1, 1,
                         kLdsBytes,
                         stream);
}
