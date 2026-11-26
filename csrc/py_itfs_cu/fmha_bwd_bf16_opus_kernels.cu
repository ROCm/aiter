// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch entry points of the OPUS attention backward (gfx950, bf16, D_QK = 192 / D_V = 128).

#include "aiter_hip_common.h"
#include "fmha_bwd_hd192_v128_bf16_opus_defs.h"
#include "torch/fmha_bwd_bf16_opus.h"

#include <ATen/hip/HIPContext.h>

#include <climits>

namespace {

// The kernels, from a code object built with the amdgpu-named-reg-pin-std toolchain
// (csrc/kernels/mla/opus/gfx950/fmha_bwd_hd192_v128_bf16.cu, build_co.sh).
constexpr const char* kCo      = "mla_opus/opus_fmha_bwd_hd192_v128_bf16.co";
constexpr const char* kPre     = "opus_fmha_bwd_v128_pre_kernel";
constexpr const char* kPreGate = "opus_fmha_bwd_v128_pre_gate_kernel";
constexpr const char* kAttn    = "opus_fmha_bwd_d192_v128_kernel";

// The attention kernel's byte offsets within a tile are 32-bit: up to 64 rows of a row stride,
// and 1 << 30 as the offset past any tile. 1 << 24 elements a row keeps both in range.
constexpr int64_t kMaxRowStride = int64_t{1} << 24;

// A [total, H, 128] bf16 tensor of the pre-pass: rows of 16-byte aligned 8-element accesses,
// strides that fit the kernel's int arguments; an output must not overlap itself.
void check_rows(const at::Tensor& t, const at::Tensor& like, const char* name, bool output)
{
    TORCH_CHECK(t.device() == like.device(), name, " must be on dout's device");
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
    TORCH_CHECK(t.dim() == 3 && t.sizes() == like.sizes(), name, " must be [total, H, 128]");
    TORCH_CHECK(t.stride(-1) == 1 && t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                name,
                " rows must be contiguous and 16-byte aligned");
    TORCH_CHECK(t.stride(0) <= INT_MAX && t.stride(1) <= INT_MAX, name, " strides exceed int32");
    TORCH_CHECK(!output || t.is_non_overlapping_and_dense(), name, " must not overlap itself");
}

// Pre-pass: the kernel with the gate, or without `gate`, `d_o` and `dgate` (all three null) the
// one that makes D alone.
void launch_pre(at::Tensor& dout,
                at::Tensor& out,
                at::Tensor* gate,
                at::Tensor* d_o,
                at::Tensor* dgate,
                at::Tensor& d)
{
    TORCH_CHECK(dout.is_cuda(), "dout must be a GPU tensor");
    TORCH_CHECK(dout.dim() == 3 && dout.size(2) == 128, "dout must be [total, H, 128]");
    TORCH_CHECK(dout.size(0) <= INT_MAX && dout.size(1) <= INT_MAX, "dout is too large");
    for(auto [t, name, output] : {std::tuple<at::Tensor*, const char*, bool>{&dout, "dout", false},
                                  {&out, "out", false},
                                  {gate, "gate", false},
                                  {d_o, "d_o", true},
                                  {dgate, "dgate", true}})
        if(t != nullptr)
            check_rows(*t, dout, name, output);
    const int total = static_cast<int>(dout.size(0));
    const int H     = static_cast<int>(dout.size(1));
    TORCH_CHECK(d.device() == dout.device() && d.scalar_type() == at::kFloat && d.dim() == 2 &&
                    d.size(0) == H && d.size(1) == total && d.stride(1) == 1 &&
                    d.stride(0) <= INT_MAX && d.is_non_overlapping_and_dense(),
                "d must be float32 [H, total], contiguous along total");

    fmha_bwd_v128_pre_kargs k{};
    k.ptr_dout      = dout.data_ptr();
    k.ptr_out       = out.data_ptr();
    k.ptr_d         = d.data_ptr<float>();
    k.total         = total;
    k.H             = H;
    k.stride_dout_n = static_cast<int>(dout.stride(0));
    k.stride_dout_h = static_cast<int>(dout.stride(1));
    k.stride_out_n  = static_cast<int>(out.stride(0));
    k.stride_out_h  = static_cast<int>(out.stride(1));
    if(gate != nullptr)
    {
        k.ptr_gate       = gate->data_ptr();
        k.ptr_do         = d_o->data_ptr();
        k.ptr_dgate      = dgate->data_ptr();
        k.stride_gate_n  = static_cast<int>(gate->stride(0));
        k.stride_gate_h  = static_cast<int>(gate->stride(1));
        k.stride_do_n    = static_cast<int>(d_o->stride(0));
        k.stride_do_h    = static_cast<int>(d_o->stride(1));
        k.stride_dgate_n = static_cast<int>(dgate->stride(0));
        k.stride_dgate_h = static_cast<int>(dgate->stride(1));
    }
    k.stride_d_h = static_cast<int>(d.stride(0));
    if(total == 0 || H == 0)
        return;

    constexpr int ROWS   = FMHA_BWD_V128_PRE_BLOCK / (128 / 8);
    const long long rows = static_cast<long long>(total) * H;
    HipDeviceGuard guard(dout.device().index());
    const int blocks = static_cast<int>((rows + ROWS - 1) / ROWS);
    size_t sz        = sizeof(k);
    // An OPUS code object (compiler output): the gfx1250 gate of shipped ASM does not apply.
    static AiterAsmKernel pre(kPre, kCo, AiterAsmKernel::SkipGfx1250Gate{});
    static AiterAsmKernel pre_gate(kPreGate, kCo, AiterAsmKernel::SkipGfx1250Gate{});
    (gate != nullptr ? pre_gate : pre)
        .launch_kernel(
            {&k, &sz, blocks, 1, 1, FMHA_BWD_V128_PRE_BLOCK, 1, 1, at::hip::getCurrentHIPStream()});
}

} // namespace

void fmha_bwd_bf16_opus_gate(at::Tensor& dout,
                             at::Tensor& out,
                             at::Tensor& gate,
                             at::Tensor& d_o,
                             at::Tensor& dgate,
                             at::Tensor& d)
{
    launch_pre(dout, out, &gate, &d_o, &dgate, d);
}

void fmha_bwd_bf16_opus_dot(at::Tensor& dout, at::Tensor& out, at::Tensor& d)
{
    launch_pre(dout, out, nullptr, nullptr, nullptr, d);
}

// The attention's backward (opus_fmha_bwd_d192_v128_kernel): the dK / dV pass and the dQ pass in
// one launch.
void fmha_bwd_bf16_opus_bwd(at::Tensor& q,
                            at::Tensor& k,
                            at::Tensor& v,
                            at::Tensor& d_o,
                            at::Tensor& lse,
                            at::Tensor& d,
                            at::Tensor& dq,
                            at::Tensor& dk,
                            at::Tensor& dv,
                            at::Tensor& seqstart_q,
                            at::Tensor& seqstart_k,
                            int max_seqlen_q,
                            int max_seqlen_k,
                            bool causal,
                            float softmax_scale)
{
    using namespace fmha_bwd_d192_v128_dkdv;
    TORCH_CHECK(q.is_cuda(), "q must be a GPU tensor");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q and k must be [total, heads, 192]");
    TORCH_CHECK(q.size(0) <= INT_MAX && q.size(1) <= INT_MAX && k.size(0) <= INT_MAX &&
                    k.size(1) <= INT_MAX,
                "q or k is too large");
    const int total_q = static_cast<int>(q.size(0));
    const int H       = static_cast<int>(q.size(1));
    const int total_k = static_cast<int>(k.size(0));
    const int H_k     = static_cast<int>(k.size(1));
    const int share   = H_k > 0 ? H / H_k : 1; // the heads that read a K / V head
    TORCH_CHECK(share * H_k == H, "q's heads must be a multiple of k's");
    // `vec` is the widest access, in elements: rows and strides must keep it aligned. A row of a
    // [total, heads, D] tensor is read through a buffer of its token stride times the rows read,
    // so the token stride must cover a row: no broadcast (stride 0) or overlapping rows. An
    // output must not overlap itself.
    auto check = [&](const at::Tensor& t,
                     at::ScalarType dt,
                     std::initializer_list<int64_t> shape,
                     int vec,
                     const char* name,
                     bool output) {
        TORCH_CHECK(t.device() == q.device(), name, " must be on q's device");
        TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
        TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " has the wrong shape");
        TORCH_CHECK(t.stride(-1) == 1, name, " must be contiguous along its last dim");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % (vec * t.element_size()) == 0,
                    name,
                    " is not aligned for its vector accesses");
        for(int i = 0; i + 1 < t.dim(); i++)
            TORCH_CHECK(t.stride(i) % vec == 0 && t.stride(i) <= INT_MAX,
                        name,
                        " strides must be multiples of ",
                        vec,
                        " that fit int32");
        if(t.dim() == 3 && t.numel() > 0)
            TORCH_CHECK(t.stride(0) >= t.size(2) && t.stride(0) <= kMaxRowStride,
                        name,
                        " token stride must be at least a row and at most ",
                        kMaxRowStride,
                        " elements");
        TORCH_CHECK(!output || t.is_non_overlapping_and_dense(), name, " must not overlap itself");
    };
    check(q, at::kBFloat16, {total_q, H, D_QK}, 8, "q", false);
    check(k, at::kBFloat16, {total_k, H_k, D_QK}, 8, "k", false);
    check(v, at::kBFloat16, {total_k, H_k, D_V}, 8, "v", false);
    check(d_o, at::kBFloat16, {total_q, H, D_V}, 8, "d_o", false);
    check(lse, at::kFloat, {H, total_q}, 1, "lse", false);
    check(d, at::kFloat, {H, total_q}, 1, "d", false);
    TORCH_CHECK(lse.strides() == d.strides(), "lse and d must share strides");
    check(dq, at::kBFloat16, {total_q, H, D_QK}, 4, "dq", true);
    check(dk, at::kBFloat16, {total_k, H, D_QK}, 4, "dk", true);
    check(dv, at::kBFloat16, {total_k, H, D_V}, 4, "dv", true);
    const int B = static_cast<int>(seqstart_q.numel()) - 1;
    TORCH_CHECK(B > 0 && seqstart_k.numel() == B + 1,
                "seqstart_q / seqstart_k must be [num_seqs + 1]");
    for(const at::Tensor* s : {&seqstart_q, &seqstart_k})
        TORCH_CHECK(s->device() == q.device() && s->scalar_type() == at::kInt && s->is_contiguous(),
                    "seqstart_q / seqstart_k must be contiguous int32 on q's device");
    TORCH_CHECK(max_seqlen_q >= 0 && max_seqlen_k >= 0,
                "max_seqlen_q and max_seqlen_k must not be negative");
    TORCH_CHECK(total_q == 0 || max_seqlen_q > 0, "max_seqlen_q must be > 0 with queries");
    TORCH_CHECK(total_k == 0 || max_seqlen_k > 0, "max_seqlen_k must be > 0 with keys");

    // Keys that no query sees have zero gradients. Within a launch the dK / dV pass writes them
    // (a sequence without queries runs one tile of zeros); with no query at all it does not run.
    if(total_q == 0 && total_k > 0)
    {
        dk.zero_();
        dv.zero_();
    }

    fmha_bwd_d192_v128_kargs a{};
    a.ptr_q          = q.data_ptr();
    a.ptr_k          = k.data_ptr();
    a.ptr_v          = v.data_ptr();
    a.ptr_do         = d_o.data_ptr();
    a.ptr_lse        = lse.data_ptr<float>();
    a.ptr_d          = d.data_ptr<float>();
    a.ptr_seqstart_q = seqstart_q.data_ptr<int>();
    a.ptr_seqstart_k = seqstart_k.data_ptr<int>();
    a.H              = H;
    a.num_seqs       = B;
    a.stride_q_n     = static_cast<int>(q.stride(0));
    a.stride_q_h     = static_cast<int>(q.stride(1));
    a.stride_k_n     = static_cast<int>(k.stride(0));
    a.stride_k_h     = static_cast<int>(k.stride(1));
    a.stride_v_n     = static_cast<int>(v.stride(0));
    a.stride_v_h     = static_cast<int>(v.stride(1));
    a.stride_do_n    = static_cast<int>(d_o.stride(0));
    a.stride_do_h    = static_cast<int>(d_o.stride(1));
    a.stride_lse_h   = static_cast<int>(lse.stride(0));
    a.softmax_scale  = softmax_scale;
    a.causal         = causal ? 1 : 0;
    // The dQ pass of queries that see no key writes zeros; the dK / dV pass has nothing to write
    // without keys.
    if(total_q > 0)
    {
        a.ptr_dq      = dq.data_ptr();
        a.stride_dq_n = static_cast<int>(dq.stride(0));
        a.stride_dq_h = static_cast<int>(dq.stride(1));
        a.n_q_blocks  = (max_seqlen_q + fmha_bwd_d192_v128_dq::BQ - 1) / fmha_bwd_d192_v128_dq::BQ;
    }
    if(total_q > 0 && total_k > 0)
    {
        a.ptr_dk      = dk.data_ptr();
        a.ptr_dv      = dv.data_ptr();
        a.stride_dk_n = static_cast<int>(dk.stride(0));
        a.stride_dk_h = static_cast<int>(dk.stride(1));
        a.stride_dv_n = static_cast<int>(dv.stride(0));
        a.stride_dv_h = static_cast<int>(dv.stride(1));
        a.n_kv_blocks = (max_seqlen_k + BK - 1) / BK;
    }
    unsigned int grid[3];
    TORCH_CHECK(fmha_bwd_d192_v128_grid(a, share, grid), "too many sequences or blocks");
    if(grid[0] == 0)
        return;

    HipDeviceGuard guard(q.device().index());
    size_t sz = sizeof(a);
    // An OPUS code object (compiler output): the gfx1250 gate of shipped ASM does not apply.
    static AiterAsmKernel attn(kAttn, kCo, AiterAsmKernel::SkipGfx1250Gate{});
    attn.launch_kernel({&a,
                        &sz,
                        static_cast<int>(grid[0]),
                        static_cast<int>(grid[1]),
                        static_cast<int>(grid[2]),
                        BLOCK,
                        1,
                        1,
                        at::hip::getCurrentHIPStream()});
}
