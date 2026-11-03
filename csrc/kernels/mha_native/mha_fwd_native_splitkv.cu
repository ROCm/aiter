#include "aiter_hip_common.h"   // aiter common HIP helpers
#include "aiter_stream.h"       // aiter::getCurrentHIPStream (torch-free)
#include "mha_native.h"
#include "mha_native_launch.h"
#include "runner/params.hpp"    // FmhaFwdParams, FmhaFwdSplitParams, FmhaFwdCombineParams, kM0, kBlockSize

// Throw-on-error HIP check (same idiom as csrc/kernels/causal_conv1d_update.cu).
// aiter_hip_common.h provides HIP_CALL / HIP_CALL_LAUNCH (abort-on-error), but the
// torch-facing path wants an exception instead, so we define HIP_CHECK locally.
#ifndef HIP_CHECK
#define HIP_CHECK(err)                                                      \
    do {                                                                    \
        hipError_t err_ = (err);                                            \
        if (err_ != hipSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("HIP error: ") + hipGetErrorString(err_) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));       \
        }                                                                   \
    } while (0)
#endif

namespace aiter {

// Convert natural-e softmax scale to the base-2 form the kernel's exp2 softmax wants.
static constexpr float kLog2e = 1.4426950408889634f;

void mha_fwd_native_splitkv(
    aiter_tensor_t& q, aiter_tensor_t& k, aiter_tensor_t& v,
    aiter_tensor_t& o, aiter_tensor_t& lse,
    aiter_tensor_t& scratch_o, aiter_tensor_t& scratch_lse,
    double softmax_scale, bool causal, bool return_lse, int64_t num_splits)
{
    AITER_CHECK(q.is_gpu() && k.is_gpu() && v.is_gpu(), "q/k/v must be HIP tensors");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && k.dtype() == AITER_DTYPE_bf16 &&
                    v.dtype() == AITER_DTYPE_bf16,
                "native splitkv is bf16 only");
    AITER_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q/k/v must be 4-D BSHD tensors");
    // Kernel indexes D contiguously and only uses batch/seqlen/head strides.
    AITER_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
                "q/k/v last dim must be contiguous");

    // aiter layout is BSHD: (B, S, H, D).
    const int B  = q.size(0);
    const int Sq = q.size(1);
    const int Hq = q.size(2);
    const int D  = q.size(3);
    const int Sk = k.size(1);
    const int Hk = k.size(2);
    AITER_CHECK(D == 64 && v.size(3) == 64, "native splitkv is D64 only");
    // GQA grouping assumes Hk divides Hq; Hk > Hq would divide by zero on device.
    AITER_CHECK(Hk > 0 && Hq % Hk == 0, "nhead_q must be a multiple of nhead_k");
    const int G = static_cast<int>(num_splits);
    AITER_CHECK(G >= 1, "num_splits must be >= 1");

    // o / lse / scratch_o / scratch_lse are pre-allocated by the caller
    // (aiter/ops/mha.py). Validate o's shape/dtype like the old out path did.
    AITER_CHECK(o.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(o.dim() == 4 && o.size(0) == B && o.size(1) == Sq &&
                    o.size(2) == Hq && o.size(3) == D,
                "out must have shape (B, Sq, Hq, D)");
    AITER_CHECK(o.stride(-1) == 1, "out last dim must be contiguous");

    FmhaFwdParams base{};
    base.q   = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    base.k   = reinterpret_cast<const __hip_bfloat16*>(k.data_ptr());
    base.v   = reinterpret_cast<const __hip_bfloat16*>(v.data_ptr());
    base.o   = reinterpret_cast<__hip_bfloat16*>(o.data_ptr());  // producers ignore base.o
    base.lse = nullptr;                                          // producers write scratch_lse
    base.seqlen_q = Sq; base.seqlen_k = Sk;
    base.nhead_q  = Hq; base.nhead_k  = Hk;
    base.scale    = static_cast<float>(softmax_scale) * kLog2e;
    // BSHD strides in ELEMENTS: token=stride(1), head=stride(2), batch=stride(0).
    base.stride_q = q.stride(1); base.nhead_stride_q = q.stride(2); base.batch_stride_q = q.stride(0);
    base.stride_k = k.stride(1); base.nhead_stride_k = k.stride(2); base.batch_stride_k = k.stride(0);
    base.stride_v = v.stride(1); base.nhead_stride_v = v.stride(2); base.batch_stride_v = v.stride(0);
    base.stride_o = o.stride(1); base.nhead_stride_o = o.stride(2); base.batch_stride_o = o.stride(0);
    base.seqstart_q = nullptr; base.seqstart_k = nullptr;

    FmhaFwdSplitParams sp{};
    sp.base        = base;
    sp.scratch_o   = static_cast<float*>(scratch_o.data_ptr());
    sp.scratch_lse = static_cast<float*>(scratch_lse.data_ptr());
    sp.num_splits  = G;
    sp.split_idx   = 0;  // vestigial: globals decode split from blockIdx.z

    HipDeviceGuard device_guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const int m_tiles = (Sq + kM0 - 1) / kM0;
    dim3 grid_prod(Hq, m_tiles, B * G);
    if (causal) launch_msk1_split(sp, grid_prod, stream);
    else        launch_msk0_split(sp, grid_prod, stream);

    FmhaFwdCombineParams cp{};
    cp.scratch_o   = static_cast<float*>(scratch_o.data_ptr());
    cp.scratch_lse = static_cast<float*>(scratch_lse.data_ptr());
    cp.o           = reinterpret_cast<__hip_bfloat16*>(o.data_ptr());
    cp.lse         = return_lse ? static_cast<float*>(lse.data_ptr()) : nullptr;
    cp.num_splits  = G;
    cp.seqlen_q    = Sq;
    cp.nhead_q     = Hq;
    cp.stride_o    = o.stride(1); cp.nhead_stride_o = o.stride(2); cp.batch_stride_o = o.stride(0);
    cp.scale       = base.scale;  // vestigial in combine (global LSE is natural-log); set for completeness
    cp.o_fp32      = nullptr;
    // combine recovers B from gridDim.z, so grid z MUST be batch (NOT batch*G).
    dim3 grid_comb(Hq, m_tiles, B);
    launch_combine(cp, grid_comb, stream);

    HIP_CHECK(hipGetLastError());
}

}  // namespace aiter
