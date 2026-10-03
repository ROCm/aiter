// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free binding + host dispatch for the fused MoE sorting kernel.
// Self-contained HIP: no CK, no Opus headers.

#include "fused_moe_sorting.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

namespace {

// One workgroup owns the whole sort, so its width is the only parallelism the
// sort gets. 1024 threads = 16 wave64s, which is what hides the scatter's
// memory latency; at 256 (4 waves) the same kernel measured 4x slower than
// Opus at M=64. Kept selectable because the best width is shape-dependent:
// a wide block buys latency hiding but costs more barrier time, and at tiny E
// most of its waves have no expert to work on.
constexpr int kBlockMax = 1024;

static int block_override()
{
    static const int v = [] {
        const char* e = getenv("AITER_FUSED_MOE_SORTING_BLOCK");
        const int n   = e ? atoi(e) : 0;
        return (n == 256 || n == 512 || n == 1024) ? n : 0;
    }();
    return v;
}

// Usable dynamic-LDS budget, read from the device at runtime and cached.
//
// This must not be a compile-time constant. Opus derives its one-shot
// threshold from a constexpr get_smem_size() evaluated in the *host* pass,
// where __gfx950__ is undefined, so it always reads 64 KB and caps one-shot at
// M<=24 even on MI355X, which actually offers 160 KB (FlyDSL reads the arch
// correctly but then clamps to 16 anyway). Querying hipGetDeviceProperties
// once and caching it keeps the threshold honest on every arch.
static size_t lds_budget_bytes()
{
    static const size_t budget = [] {
        hipDevice_t dev;
        hipDeviceProp_t prop;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&prop, dev));
        return static_cast<size_t>(prop.sharedMemPerBlock);
    }();
    return budget;
}

// AITER_CHECK aborts the process unless throwing is enabled for the thread.
// An out-of-budget shape is a routing decision, not a fatal condition -- the
// caller is meant to fall back to Opus -- so the guard has to be recoverable.
struct AiterThrowGuard
{
    bool prev;
    AiterThrowGuard() : prev(aiter_detail::g_aiter_can_throw)
    { aiter_detail::g_aiter_can_throw = true; }
    ~AiterThrowGuard() { aiter_detail::g_aiter_can_throw = prev; }
};

// Upper bound on the tokens this op will take, independent of whether the LDS
// budget would stretch further.
//
// The sort runs in one workgroup, i.e. on one CU, trading the GPU's width for
// a single launch and no HBM round trip. Where that stops paying is not a
// tuning accident, it is in the cost model: ranking assignments inside an
// expert deterministically means scanning a bitmask of E * ceil(M*topk/64)
// words, so this kernel carries an O(E*M*topk) term on one CU while the rivals
// carry O(M*topk) spread over many. The fixed savings win at small M and that
// term wins at large M; there is a crossover, and no amount of tuning inside
// this design removes it.
//
// Measured on an idle GPU, all candidates in one interleaved trace, min of
// ~600 dispatches, three independent repeats. Against the fastest of
// Opus/CK/FlyDSL:
//   M=64   1.13x 1.15x 1.15x
//   M=80   1.09x 1.04x 1.11x
//   M=96   1.04x 1.06x 1.00x   <- one repeat only reached parity
//   M=112  1.01x 0.98x 0.99x   <- crossover
// This kernel is the stable side of that comparison (M=80 and M=112 returned
// identical times in all three repeats); the spread comes from the rivals. So
// the cap is 80, the last point that wins in every repeat, rather than 96,
// where a run can land on parity. Past it the op declines and the caller stays
// on Opus/CK -- also where the campaign scopes prefill. LDS alone would admit
// M=2048; this is a separate, deliberate gate. Raise it with
// AITER_FUSED_MOE_SORTING_MAX_M to re-measure.
static int max_fused_tokens()
{
    static const int v = [] {
        const char* e = getenv("AITER_FUSED_MOE_SORTING_MAX_M");
        const int n   = e ? atoi(e) : 0;
        return n > 0 ? n : 80;
    }();
    return v;
}

// Cap for the fused-router path. With multi-block routing (see
// fused_moe_sorting_body) the router phase is flat in M -- ~2.9 us at both
// M=16 and M=64 -- so this path now beats aiter's router plus the sort-only
// kernel across the whole supported range (1.22-1.34x, idle GPU, interleaved
// trace) and shares the sort-only cap. Overridable via
// AITER_FUSED_MOE_SORTING_TOPK_MAX_M.
static int max_fused_topk_tokens()
{
    static const int v = [] {
        const char* e = getenv("AITER_FUSED_MOE_SORTING_TOPK_MAX_M");
        const int n   = e ? atoi(e) : 0;
        return n > 0 ? n : max_fused_tokens();
    }();
    return v;
}

// Waves per router block, i.e. tokens per router block, in multi-block routing.
//
// Fewer waves per block means fewer chains sharing a SIMD (the per-token
// arg-max chain runs at full rate with one wave per SIMD) but more router
// blocks, and blocks are dispatched in order, so the last one starts later.
// Measured clean on an idle GPU, fused-router kernel, us:
//     M      16     32     48     64     80
//     RW=4   5.92   6.24   6.88   7.08   7.60
//     RW=8   5.64   6.04   6.32   6.52   7.84
// Eight wins through M=64 (fewer blocks, less dispatch skew, two chains per
// SIMD still cheap enough); at 80 the extra chain contention overtakes it.
// Selected from the static capacity, so a captured graph replays unchanged.
static int router_waves_for(int num_tokens)
{
    if(const char* e = getenv("AITER_FUSED_MOE_SORTING_ROUTER_WAVES"))
    {
        const int n = atoi(e);
        if(n == 4 || n == 8)
            return n;
    }
    return num_tokens <= 64 ? 8 : 4;
}

struct LaunchPlan
{
    bool supported;
    bool stage_weights;
    size_t smem_bytes;
};

// The fused-router path has to stage weights -- it produces them in-kernel and
// there is nowhere else to put them. The sort-only path does not: since the
// scatter walks assignments in index order, reading topk_weights[idx] straight
// from global is coalesced, so staging buys nothing and costs LDS that is
// better spent on reach.
static LaunchPlan plan_launch(int num_tokens, int num_experts, int topk, int router_experts = 0)
{
    const int cap = router_experts > 0 ? max_fused_topk_tokens() : max_fused_tokens();
    if(num_tokens > cap)
        return LaunchPlan{false, false, 0};

    const size_t budget   = lds_budget_bytes();
    const bool must_stage = router_experts > 0;
    const auto s          = aiter::FusedMoeSortingSmem::make(
        num_tokens, num_experts, topk, kBlockMax, must_stage, router_experts);
    if(s.bytes <= budget)
        return LaunchPlan{true, must_stage, s.bytes};
    return LaunchPlan{false, false, 0};
}

template <int BLOCK, bool STAGE_W>
static void launch(const LaunchPlan& plan,
                   dim3 grid,
                   hipStream_t stream,
                   const int32_t* topk_ids,
                   const float* topk_weights,
                   const int32_t* expert_mask,
                   const int32_t* num_local_tokens,
                   int32_t* sorted_ids,
                   float* sorted_weights,
                   int32_t* sorted_expert_ids,
                   int32_t* num_valid_ids,
                   void* moe_buf,
                   int num_tokens,
                   int topk,
                   int num_experts,
                   int unit_size,
                   size_t moe_buf_bytes,
                   int zero_blocks)
{
    // Dynamic LDS above 64 KB needs the opt-in attribute even where the device
    // reports a larger budget. Raised once per instantiation to the whole
    // budget rather than per launch: the call is not free, and this op is
    // dispatched 75 times per decode step.
    static const bool once = [] {
        auto* f = reinterpret_cast<const void*>(&aiter::fused_moe_sorting_bitmask<BLOCK, STAGE_W>);
        (void)hipFuncSetAttribute(
            f, hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(lds_budget_bytes()));
        return true;
    }();
    (void)once;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(aiter::fused_moe_sorting_bitmask<BLOCK, STAGE_W>),
                       grid,
                       dim3(BLOCK),
                       plan.smem_bytes,
                       stream,
                       topk_ids,
                       topk_weights,
                       expert_mask,
                       num_local_tokens,
                       sorted_ids,
                       sorted_weights,
                       sorted_expert_ids,
                       num_valid_ids,
                       moe_buf,
                       num_tokens,
                       topk,
                       num_experts,
                       unit_size,
                       moe_buf_bytes,
                       zero_blocks);
}

} // namespace

int fused_moe_sorting_get_workspace_size(int tokens, int num_experts, int topk, int unit_size)
{
    // The whole sort lives in LDS, so there is no global workspace to hand in.
    // Kept for signature parity with moe_sorting_opus_get_workspace_size, and
    // so callers can allocate uniformly across backends.
    (void)tokens;
    (void)num_experts;
    (void)topk;
    (void)unit_size;
    return 0;
}

bool fused_moe_sorting_is_supported(int tokens, int num_experts, int topk, int unit_size)
{
    if(tokens <= 0 || num_experts <= 0 || topk <= 0 || unit_size <= 0)
        return false;
    // The slot is packed into the top 8 bits of the id, and 3 bitplanes encode
    // it in the 4-plane layout.
    if(topk > 8)
        return false;
    return plan_launch(tokens, num_experts, topk).supported;
}

void fused_moe_sorting_fwd(aiter_tensor_t& topk_ids,
                           aiter_tensor_t& topk_weights,
                           aiter_tensor_t& sorted_token_ids,
                           aiter_tensor_t& sorted_weights,
                           aiter_tensor_t& sorted_expert_ids,
                           aiter_tensor_t& num_valid_ids,
                           aiter_tensor_t& moe_buf,
                           int num_experts,
                           int unit_size,
                           std::optional<aiter_tensor_t> local_expert_mask,
                           std::optional<aiter_tensor_t> num_local_tokens)
{
    const AiterThrowGuard throw_guard;

    AITER_CHECK(topk_ids.dtype() == AITER_DTYPE_i32, "topk_ids must be int32");
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32, "topk_weights must be fp32");
    AITER_CHECK(topk_ids.dim() == 2, "topk_ids must be [tokens, topk]");
    AITER_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");

    // Static capacity, never a device-side value: every host-side launch
    // decision has to be identical between CUDA-graph capture and replay, so
    // num_local_tokens is read on-device inside the kernel and never with
    // .item() here.
    const int num_tokens = static_cast<int>(topk_ids.size(0));
    const int topk       = static_cast<int>(topk_ids.size(1));

    const auto plan = plan_launch(num_tokens, num_experts, topk);
    AITER_CHECK(plan.supported,
                "fused_moe_sorting: shape outside the fused range (token cap or "
                "LDS budget); caller should fall back to moe_sorting_opus");
    AITER_CHECK(topk <= 8, "fused_moe_sorting: topk > 8 unsupported");

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const size_t moe_buf_bytes = moe_buf.numel() * moe_buf.element_size();

    // Narrower blocks only make sense while there is little per-expert work to
    // spread; the override exists so the choice can be swept without a rebuild.
    int block = block_override();
    if(block == 0)
        block = 1024;

    // Block 0 sorts; the rest zero moe_buf alongside it. One 16 B store per
    // thread, so the zero pass spreads over many CUs instead of queueing on a
    // few: block 0's sort is the long pole and the zeroing has to disappear
    // behind it, not extend the kernel.
    int zero_blocks = 0;
    if(moe_buf_bytes > 0)
    {
        const size_t n_vec   = moe_buf_bytes / sizeof(uint4);
        const size_t per_blk = static_cast<size_t>(block);
        zero_blocks = static_cast<int>(std::min<size_t>((n_vec + per_blk - 1) / per_blk, 2048));
        zero_blocks = std::max(zero_blocks, 1);
    }
    const dim3 grid(1 + zero_blocks);

    const auto* mask_ptr = local_expert_mask.has_value()
                               ? static_cast<const int32_t*>(local_expert_mask.value().data_ptr())
                               : nullptr;
    const auto* nlt_ptr  = num_local_tokens.has_value()
                               ? static_cast<const int32_t*>(num_local_tokens.value().data_ptr())
                               : nullptr;

    auto dispatch_b = [&](auto block_tag, auto stage_tag) {
        launch<decltype(block_tag)::value, decltype(stage_tag)::value>(
            plan,
            grid,
            stream,
            static_cast<const int32_t*>(topk_ids.data_ptr()),
            static_cast<const float*>(topk_weights.data_ptr()),
            mask_ptr,
            nlt_ptr,
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(num_valid_ids.data_ptr()),
            moe_buf.data_ptr(),
            num_tokens,
            topk,
            num_experts,
            unit_size,
            moe_buf_bytes,
            zero_blocks);
    };

    auto dispatch = [&](auto block_tag) {
        if(plan.stage_weights)
            dispatch_b(block_tag, std::true_type{});
        else
            dispatch_b(block_tag, std::false_type{});
    };

    if(block == 256)
        dispatch(std::integral_constant<int, 256>{});
    else if(block == 512)
        dispatch(std::integral_constant<int, 512>{});
    else
        dispatch(std::integral_constant<int, 1024>{});
}

// ---------------------------------------------------------------------------
// F1-b: sorting with the router folded in.

namespace {

template <int BLOCK, int RNREG, int RW>
static void launch_topk(const LaunchPlan& plan,
                        dim3 grid,
                        hipStream_t stream,
                        const int32_t* expert_mask,
                        const int32_t* num_local_tokens,
                        int32_t* sorted_ids,
                        float* sorted_weights,
                        int32_t* sorted_expert_ids,
                        int32_t* num_valid_ids,
                        void* moe_buf,
                        int num_tokens,
                        int topk,
                        int num_experts,
                        int unit_size,
                        size_t moe_buf_bytes,
                        int zero_blocks,
                        aiter::FusedRouterArgs router)
{
    static const bool once = [] {
        auto* f = reinterpret_cast<const void*>(
            &aiter::fused_moe_sorting_topk_bitmask<BLOCK, true, RNREG, RW>);
        (void)hipFuncSetAttribute(
            f, hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(lds_budget_bytes()));
        return true;
    }();
    (void)once;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(aiter::fused_moe_sorting_topk_bitmask<BLOCK, true, RNREG, RW>),
        grid,
        dim3(BLOCK),
        plan.smem_bytes,
        stream,
        expert_mask,
        num_local_tokens,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_tokens,
        topk,
        num_experts,
        unit_size,
        moe_buf_bytes,
        zero_blocks,
        router);
}

} // namespace

bool fused_moe_sorting_topk_is_supported(
    int tokens, int num_experts, int topk, int unit_size, int router_experts)
{
    if(tokens <= 0 || num_experts <= 0 || topk <= 0 || unit_size <= 0 || router_experts <= 0)
        return false;
    if(topk > 8 || topk > 64)
        return false; // slot field is 8 bits; lane `k` carries result k
    return plan_launch(tokens, num_experts, topk, router_experts).supported;
}

void fused_moe_sorting_topk_fwd(aiter_tensor_t& gating_output,
                                aiter_tensor_t& sorted_token_ids,
                                aiter_tensor_t& sorted_weights,
                                aiter_tensor_t& sorted_expert_ids,
                                aiter_tensor_t& num_valid_ids,
                                aiter_tensor_t& moe_buf,
                                int num_experts,
                                int topk,
                                int unit_size,
                                bool need_renorm,
                                bool is_softmax,
                                float routed_scaling_factor,
                                std::optional<aiter_tensor_t> local_expert_mask,
                                std::optional<aiter_tensor_t> num_local_tokens,
                                std::optional<aiter_tensor_t> sync)
{
    const AiterThrowGuard throw_guard;

    AITER_CHECK(gating_output.dim() == 2, "gating_output must be [tokens, router_experts]");
    AITER_CHECK(gating_output.is_contiguous(), "gating_output must be contiguous");
    const bool is_fp32 = gating_output.dtype() == AITER_DTYPE_fp32;
    AITER_CHECK(is_fp32 || gating_output.dtype() == AITER_DTYPE_bf16,
                "gating_output must be fp32 or bf16");

    const int num_tokens     = static_cast<int>(gating_output.size(0));
    const int router_experts = static_cast<int>(gating_output.size(1));

    const auto plan = plan_launch(num_tokens, num_experts, topk, router_experts);
    AITER_CHECK(plan.supported,
                "fused_moe_sorting_topk: shape outside the fused range (token cap "
                "or LDS budget); caller should fall back to grouped_topk + "
                "moe_sorting_opus");
    AITER_CHECK(topk <= 8, "fused_moe_sorting_topk: topk > 8 unsupported");

    HipDeviceGuard device_guard(gating_output.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const size_t moe_buf_bytes = moe_buf.numel() * moe_buf.element_size();

    int block = block_override();
    if(block == 0)
        block = 1024;

    int zero_blocks = 0;
    if(moe_buf_bytes > 0)
    {
        const size_t n_vec   = moe_buf_bytes / sizeof(uint4);
        const size_t per_blk = static_cast<size_t>(block);
        zero_blocks = static_cast<int>(std::min<size_t>((n_vec + per_blk - 1) / per_blk, 2048));
        zero_blocks = std::max(zero_blocks, 1);
    }
    // Register-resident router when the shape allows it: sigmoid scoring,
    // experts a multiple of the vector width, at most 64 (NREG 4) or 128
    // (NREG 8) vectors per lane, and a 16 B-aligned gating tile for the float4
    // loads. Everything else takes the LDS router (RNREG 0).
    int rnreg = 0;
    if(!is_softmax && router_experts % 4 == 0 &&
       (reinterpret_cast<uintptr_t>(gating_output.data_ptr()) % 16) == 0)
    {
        const int nvec = router_experts / 4;
        rnreg          = nvec <= 64 ? 4 : (nvec <= 128 ? 8 : 0);
    }
    if(const char* e = getenv("AITER_FUSED_MOE_SORTING_ROUTER_LDS"); e && atoi(e) == 1)
        rnreg = 0; // developer knob: force the LDS router for A/B

    // Multi-block routing needs the register router and a caller-owned sync
    // word. Router blocks are sized from the static capacity so the launch is
    // graph-replayable; zero blocks are raised to cover them.
    int32_t* sync_ptr = nullptr;
    int router_blocks = 0;
    const int rw      = router_waves_for(num_tokens);
    if(sync.has_value() && rnreg > 0)
    {
        AITER_CHECK(sync.value().dtype() == AITER_DTYPE_i32 && sync.value().numel() >= 1,
                    "sync must be int32[>=1], zero-initialised by the caller");
        sync_ptr      = static_cast<int32_t*>(sync.value().data_ptr());
        router_blocks = (num_tokens + rw - 1) / rw;
        // Router block r lives at blockIdx r * STRIDE; the grid must reach the last one.
        zero_blocks = std::max(zero_blocks, (router_blocks - 1) * FUSED_ROUTER_BLOCK_STRIDE);
    }
    const dim3 grid(1 + zero_blocks);

    aiter::FusedRouterArgs router{gating_output.data_ptr(),
                                  is_fp32,
                                  router_experts,
                                  need_renorm,
                                  is_softmax,
                                  routed_scaling_factor,
                                  sync_ptr,
                                  router_blocks};

    const auto* mask_ptr = local_expert_mask.has_value()
                               ? static_cast<const int32_t*>(local_expert_mask.value().data_ptr())
                               : nullptr;
    const auto* nlt_ptr  = num_local_tokens.has_value()
                               ? static_cast<const int32_t*>(num_local_tokens.value().data_ptr())
                               : nullptr;

    auto go = [&](auto block_tag, auto rn_tag, auto rw_tag) {
        launch_topk<decltype(block_tag)::value, decltype(rn_tag)::value, decltype(rw_tag)::value>(
            plan,
            grid,
            stream,
            mask_ptr,
            nlt_ptr,
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<float*>(sorted_weights.data_ptr()),
            static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(num_valid_ids.data_ptr()),
            moe_buf.data_ptr(),
            num_tokens,
            topk,
            num_experts,
            unit_size,
            moe_buf_bytes,
            zero_blocks,
            router);
    };

    auto by_rw = [&](auto block_tag, auto rn_tag) {
        if(rw == 8)
            go(block_tag, rn_tag, std::integral_constant<int, 8>{});
        else
            go(block_tag, rn_tag, std::integral_constant<int, 4>{});
    };
    auto by_planes = [&](auto block_tag) {
        if(rnreg == 4)
            by_rw(block_tag, std::integral_constant<int, 4>{});
        else if(rnreg == 8)
            by_rw(block_tag, std::integral_constant<int, 8>{});
        else
            by_rw(block_tag, std::integral_constant<int, 0>{});
    };

    if(block == 256)
        by_planes(std::integral_constant<int, 256>{});
    else if(block == 512)
        by_planes(std::integral_constant<int, 512>{});
    else
        by_planes(std::integral_constant<int, 1024>{});
}
