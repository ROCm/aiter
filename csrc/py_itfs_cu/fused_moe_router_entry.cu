// Host entry for the fused MoE routing kernel (module_fused_moe_router).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <type_traits>

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
#include "aiter_hip_common.h"
#include "aiter_stream.h"

// Include dirs are csrc/include, not csrc, so reach the kernel relatively.
#include "../kernels/fused_moe_router.cu"

namespace {
// Workspace layout, owned by the caller (see get_fused_moe_router_workspace):
//   [0, kTokScaleOffset)  barrier semaphore, one u32
//   [kTokScaleOffset, .)  tok_scale, tokens rows of kMaxScaleNPad bytes
// The semaphore is self-resetting, so the caller zeroes it once at allocation
// and no launch ever memsets it; a per-call stream memset would cost more than
// the barrier it guards.
//
// scaleN_pad = pad8(cols / group_size) <= 256: cols is pinned to 4096 and
// group_size is a multiple of TD == 16. Rows are sized to that bound rather
// than the call's own scaleN_pad, so a quant-config change cannot resize the
// workspace either.
static constexpr int kMaxScaleNPad   = 256;
static constexpr int kTokScaleOffset = 256; // keeps tok_scale 256B-aligned
} // namespace

// Bytes the caller must allocate to serve every M up to max_tokens. Exposed so
// the Python side sizes the workspace without duplicating the layout.
int64_t fused_moe_router_workspace_size(int64_t max_tokens)
{
    TORCH_CHECK(max_tokens > 0,
                "fused_moe_router_workspace_size: max_tokens must be positive, got ",
                max_tokens);
    return kTokScaleOffset + max_tokens * (int64_t)kMaxScaleNPad;
}

void fused_moe_router_impl(
    torch::Tensor& gating, torch::Tensor& bias, torch::Tensor& hidden,
    torch::Tensor& topk_ids, torch::Tensor& topk_weights,
    torch::Tensor& sorted_ids, torch::Tensor& sorted_weights,
    torch::Tensor& sorted_expert_ids, torch::Tensor& num_valid_ids,
    torch::Tensor& out_fp4, torch::Tensor& out_scale,
    int64_t num_experts, int64_t topk, int64_t unit_size, int64_t group_size,
    bool need_renorm, double routed_scaling_factor,
    // Caller-owned scratch, >= fused_moe_router_workspace_size(M) bytes, with
    // its first 4 bytes zeroed once at allocation. Caller-owned so that a
    // CUDA-graph capture, which bakes this pointer into the replayed launch,
    // cannot have it reallocated out from under it.
    torch::Tensor& workspace,
    // [num_experts] int32, nonzero iff this rank owns the expert; None = no EP
    std::optional<torch::Tensor> expert_mask,
    // [M, model_dim] bf16 stage2 accumulator, zeroed by the kernel (stage2
    // accumulates atomically). None = caller zeroed it already.
    std::optional<torch::Tensor> moe_buf,
    // Fused shared experts appended after the routed ids, 0 = none. Each token
    // gives every shared expert shared_expert_weight. Under EP the shared
    // weights are replicated on every rank while every rank sees every token,
    // so token ownership round-robins over ep_size to keep the post-MoE
    // all-reduce from summing ep_size copies; ep_size == 1 is the non-EP case.
    int64_t num_fused_shared_experts,
    double  shared_expert_weight,
    int64_t ep_rank,
    int64_t ep_size)
{
    using namespace aiter::fmr;
    opus::bf16_t* moe_buf_ptr = nullptr;
    int           moe_buf_elems = 0;
    if(moe_buf.has_value() && moe_buf->numel() > 0)
    {
        TORCH_CHECK(moe_buf->scalar_type() == at::kBFloat16,
                    "fused_moe_router_impl: moe_buf must be bf16");
        TORCH_CHECK(moe_buf->is_contiguous(),
                    "fused_moe_router_impl: moe_buf must be contiguous");
        moe_buf_ptr   = reinterpret_cast<opus::bf16_t*>(moe_buf->data_ptr());
        moe_buf_elems = (int)moe_buf->numel();
    }
    const int M    = gating.size(0);
    const int E    = num_experts;
    const int cols = hidden.size(1);
    // TD is the MXFP4 quant's per-thread vector width and is fixed; the block size follows
    // the model dim so one thread still covers exactly one vector of the row in a single
    // pass (cols == BlockSize * TD). Pinning BlockSize instead would pin the model dim,
    // which is what restricted this entry to 4096.
    constexpr int TD        = 16;
    const int     BlockSize = cols / TD;
    // Instantiated shared-expert counts. Every model that fuses shared experts
    // today has exactly one; a wider ladder is dead template instantiations.
    constexpr int kMaxShared = 1;
    const int     n_shared   = (int)num_fused_shared_experts;
    TORCH_CHECK(n_shared >= 0 && n_shared <= kMaxShared,
                "fused_moe_router_impl: num_fused_shared_experts must be in 0..",
                kMaxShared, ", got ", n_shared);
    // ep_size is the round-robin modulus and ep_rank picks this rank's residue,
    // so a bad pair silently gives every token to one rank or to none.
    TORCH_CHECK(ep_size >= 1 && ep_rank >= 0 && ep_rank < ep_size,
                "fused_moe_router_impl: need 0 <= ep_rank < ep_size, got ep_rank=",
                ep_rank, " ep_size=", ep_size);
    // Dims this entry is instantiated for: 4096 (BlockSize 256) and 6144 (BlockSize 384).
    // Both are whole numbers of waves and satisfy E <= 2*BlockSize below. Serving another
    // dim is one more tag in dispatch_block -- nothing in the kernel body is dim-specific.
    TORCH_CHECK(cols % TD == 0 && (BlockSize == 256 || BlockSize == 384),
                "fused_moe_router_impl: model dim ", cols,
                " is not instantiated; served dims are 4096 and 6144");
    // The histogram, the scan and the local-id table are all indexed by emitted
    // expert id, so it is the full slot count -- not the routed count -- that
    // has to fit s_scan's 2*BlockSize slots.
    const bool ep    = expert_mask.has_value() && expert_mask->defined();
    const int  E_tot = aiter::fmr::expert_slots(E, n_shared, ep);
    // Non-owners park the shared row on the sentinel slot, which exists only
    // when a mask does. Without one there is nothing masking it out and no slot
    // to hold it, so the histogram would take an out-of-range atomicAdd.
    TORCH_CHECK(n_shared == 0 || ep || ep_size == 1,
                "fused_moe_router_impl: ep_size=", ep_size,
                " needs an expert_mask (the shared sentinel slot lives in it)");
    // Routed experts only: the pair scan reaches 2*BlockSize slots, and the
    // fused shared tail past that is filled serially rather than scanned.
    TORCH_CHECK(E <= 2 * BlockSize,
                "fused_moe_router_impl: num_experts must be <= ", 2 * BlockSize,
                ", got ", E);
    // A partial group would make the abs-max reduction span the wrong lanes.
    TORCH_CHECK(group_size % TD == 0,
                "fused_moe_router_impl: group_size must be a multiple of ", TD,
                ", got ", group_size);
    // The phase-3 scatter's hoisted swizzle table is only complete on an aligned
    // column span; a partial span leaves trailing scale columns holding stale
    // allocator memory that is read back as e8m0 exponents. Implied by
    // cols == BlockSize * TD, checked so the coupling cannot be lost silently.
    TORCH_CHECK((cols + group_size - 1) / group_size % aiter::fmr::kScalesPerThread == 0,
                "fused_moe_router_impl: scales per row (ceil(cols/group_size)) must be a "
                "multiple of ", aiter::fmr::kScalesPerThread, ", got ",
                (cols + group_size - 1) / group_size, " for cols=", cols,
                " group_size=", group_size);
    // The kernel shifts instead of dividing by unit_size. Every AITER
    // block_size is a power of two, so check rather than carry a fallback.
    TORCH_CHECK(unit_size > 0 && (unit_size & (unit_size - 1)) == 0,
                "fused_moe_router_impl: unit_size must be a power of two, got ", unit_size);
    // Phase 1 selects within one wave and the shared experts take the lanes
    // just past topk, so the two together must fit those 64 lanes.
    const int topk_total = (int)topk + n_shared;
    TORCH_CHECK(topk > 0 && topk_total <= 64,
                "fused_moe_router_impl: topk + fused shared experts must be in "
                "1..64 (phase 1 selects within one wave), got ", topk_total);
    // Rounds past E elect a sentinel lane (expert == INT_MAX), which the weight
    // recompute would then use to index the gating row.
    TORCH_CHECK(topk <= E,
                "fused_moe_router_impl: topk must be <= num_experts, got topk=", topk,
                " num_experts=", E);
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8 && workspace.is_contiguous(),
                "fused_moe_router_impl: workspace must be contiguous uint8");
    TORCH_CHECK(workspace.device() == gating.device(),
                "fused_moe_router_impl: workspace is on ", workspace.device(),
                " but the inputs are on ", gating.device());
    TORCH_CHECK(workspace.numel() >= fused_moe_router_workspace_size(M),
                "fused_moe_router_impl: workspace has ", workspace.numel(),
                " B, need ", fused_moe_router_workspace_size(M), " B for num_tokens=", M,
                "; size it with fused_moe_router_workspace_size");

    // The mask is indexed by global expert id, over the routed experts, the
    // fused shared slots and, when both EP and shared fusion are on, the
    // trailing sentinel -- exactly vLLM's global_num_experts + shared + 1
    // layout (expert_map_manager.py), whose sentinel is zero and whose shared
    // slots are owned. Local ids still match the stock path because local(e) is
    // an *exclusive* prefix.
    const int* mask_ptr = nullptr;
    if(ep)
    {
        TORCH_CHECK(expert_mask->numel() >= E_tot,
                    "fused_moe_router_impl: expert_mask must have at least ", E_tot,
                    " entries (num_experts + fused shared + sentinel), got ",
                    expert_mask->numel());
        TORCH_CHECK(expert_mask->scalar_type() == torch::kInt32,
                    "fused_moe_router_impl: expert_mask must be int32");
        TORCH_CHECK(expert_mask->is_contiguous(),
                    "fused_moe_router_impl: expert_mask must be contiguous");
        TORCH_CHECK(expert_mask->device() == gating.device(),
                    "fused_moe_router_impl: expert_mask is on ",
                    expert_mask->device(), " but the inputs are on ",
                    gating.device());
        mask_ptr = expert_mask->data_ptr<int>();
    }

    // Both the stream and the properties must come from the device the tensors
    // are on, not whichever one is ambient.
    const HipDeviceGuard device_guard(gating.device().index());
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Grid width multiplies barrier cost, so size it to the work and keep it under num_cu
    // so every block stays resident.
    //
    // The work is phase 3's PADDED SORTED ROWS, not the token count. Every expert the
    // routing occupies is padded up to a whole unit_size, so the row count is
    // occupied_experts * unit_size -- three to four times the token count on real routing,
    // and the quantity phase 3 divides across the grid. Sizing from M left the row count
    // serialized: at 2 816 rows over 16 blocks phase 3 ran 21.6 us, and over 64 blocks 7.8.
    //
    // The occupancy is not visible here -- the kernel produces it in phase 2, after the grid
    // is fixed -- but it is bounded by the pick count and by the slot count, and for
    // independent picks its expectation is the coupon-collector value. Real routing is more
    // concentrated than independent, so this estimate runs high and sizes the grid
    // generously; that is the safe direction, because an over-wide grid costs barrier width
    // while an under-wide one serializes phase 3.
    const int    fmr_picks    = M * topk_total;
    const double fmr_p_unhit  = std::pow(1.0 - 1.0 / (double)E_tot, (double)fmr_picks);
    const int    fmr_est_occ  = (int)std::ceil((double)E_tot * (1.0 - fmr_p_unhit));
    const int    fmr_est_rows = std::max(1, fmr_est_occ) * (int)unit_size;
    // Rows per block, the policy's one number, read off the grid x M sweep as the largest
    // value whose measured body stays within 5% of the best grid at every measured M.
    constexpr int kRowsPerBlock = 20;
    // The floor is phase 3's, not phase 1's: phase 3's rows do not shrink with M.
    int GRID   = std::max(16, (fmr_est_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    int dev_id = 0;
    HIP_CALL(hipGetDevice(&dev_id));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, dev_id));
    // Half the CUs is where widening stops repaying. The grid barrier's cost grows with the
    // block count, and past this it overtakes the parallelism it buys: in the sweep, 256
    // blocks is worse than 128 at every row count measured, while 128 is best or within
    // noise of best at every row count above one unit per block.
    GRID = std::min(GRID, std::max(16, prop.multiProcessorCount / 2));
    GRID = std::max(1, std::min(GRID, prop.multiProcessorCount));

    const int total_routed_rows = M * topk_total;
    const int max_blocks        = (int)sorted_expert_ids.numel();
    const int max_tokens        = (int)sorted_ids.numel();
    const int scaleN_valid      = (cols + group_size - 1) / group_size;
    const int scaleN_pad =
        ((scaleN_valid + kScalesPerThread - 1) / kScalesPerThread) * kScalesPerThread;

    const size_t shmem =
        aiter::fmr::LdsLayout(BlockSize, total_routed_rows, E_tot, mask_ptr != nullptr).bytes;

    // LDS grows as ~16 * M * topk, so a large enough M overruns the per-block
    // budget and the launch fails with an opaque hipErrorInvalidValue.
    TORCH_CHECK(shmem <= (size_t)prop.sharedMemPerBlock,
                "fused_moe_router_impl: shared memory request ", shmem,
                " B exceeds the per-block limit of ", prop.sharedMemPerBlock,
                " B (num_tokens=", M, ", topk=", topk, ", num_experts=", E,
                "); reduce the token count");

    auto* ws_base      = reinterpret_cast<uint8_t*>(workspace.data_ptr());
    auto* ws_sem       = reinterpret_cast<unsigned int*>(ws_base);
    auto* ws_tok_scale = ws_base + kTokScaleOffset;

    // Every buffer below is reached through reinterpret_cast, so nothing else
    // would catch a wrong dtype: fp16 read as bf16 mis-routes silently. The
    // kernel indexes with computed strides, so a non-contiguous tensor reads
    // the wrong elements rather than failing.
    auto check_bf16 = [](const torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16,
                    "fused_moe_router_impl: ", name, " must be bf16, got ",
                    t.scalar_type());
        TORCH_CHECK(t.is_contiguous(),
                    "fused_moe_router_impl: ", name, " must be contiguous");
    };
    // The activation must be bf16 -- it is what gets quantized. The router logits need not
    // be: sglang#35055 makes the ROCm router GEMM emit fp32 to match CUDA, and coercing
    // them back to bf16 here would reintroduce the rounding that fix removes. Dispatched
    // on the real dtype, like the bias.
    TORCH_CHECK(gating.scalar_type() == at::kFloat || gating.scalar_type() == at::kBFloat16,
                "fused_moe_router_impl: gating must be float32 or bfloat16, got ",
                gating.scalar_type());
    TORCH_CHECK(gating.is_contiguous(), "fused_moe_router_impl: gating must be contiguous");
    check_bf16(hidden, "hidden");
    // The rest are reached through data_ptr<T>, which checks the dtype but not
    // the layout, or are cast to a byte type where only the layout matters.
    for(const auto& p : {std::make_pair(&out_fp4, "out_fp4"),
                         std::make_pair(&out_scale, "out_scale"),
                         std::make_pair(&bias, "bias"),
                         std::make_pair(&topk_ids, "topk_ids"),
                         std::make_pair(&topk_weights, "topk_weights"),
                         std::make_pair(&sorted_ids, "sorted_ids"),
                         std::make_pair(&sorted_weights, "sorted_weights"),
                         std::make_pair(&sorted_expert_ids, "sorted_expert_ids"),
                         std::make_pair(&num_valid_ids, "num_valid_ids")})
        TORCH_CHECK(p.first->is_contiguous(),
                    "fused_moe_router_impl: ", p.second, " must be contiguous");
    // The shared rows extend each token's stride, and the kernel indexes these
    // with the widened stride; a caller that sized them [M, topk] would have
    // every token past the first write out of bounds.
    for(const auto& p : {std::make_pair(&topk_ids, "topk_ids"),
                         std::make_pair(&topk_weights, "topk_weights")})
        TORCH_CHECK(p.first->numel() >= (int64_t)M * topk_total,
                    "fused_moe_router_impl: ", p.second, " has ", p.first->numel(),
                    " elements, need M * (topk + fused shared) = ",
                    (int64_t)M * topk_total);

    const opus::bf16_t* h = reinterpret_cast<const opus::bf16_t*>(hidden.data_ptr());
    const bool gating_is_f32 = gating.scalar_type() == at::kFloat;

    // The bias is not necessarily bf16 and, unlike the stock
    // biased_grouped_topk wrapper, this path does not coerce it -- reading fp32
    // as bf16 silently mis-routes. Dispatch on the real dtype rather than
    // converting on the host: the bias is only read in phase 1's top-k.
    TORCH_CHECK(bias.scalar_type() == at::kFloat ||
                    bias.scalar_type() == at::kBFloat16,
                "fused_moe_router_impl: correction bias must be float32 or "
                "bfloat16, got ", bias.scalar_type());
    const bool bias_is_f32 = bias.scalar_type() == at::kFloat;

    // One body, instantiated per (bias dtype, shared-expert count).
    auto launch_all = [&](auto bias_tag, auto nshared_tag, auto bs_tag, auto gating_tag) {
        using DB                = decltype(bias_tag);
        using DG                = decltype(gating_tag);
        const DG* g = reinterpret_cast<const DG*>(gating.data_ptr());
        constexpr int NSHARED   = decltype(nshared_tag)::value;
        // Compile-time twin of the runtime BlockSize checked above; the kernel is
        // templated on it, so the two must agree and dispatch_block is what makes them.
        constexpr int kBlock    = decltype(bs_tag)::value;
        const DB* b = reinterpret_cast<const DB*>(bias.data_ptr());
        auto* kern = aiter::fmr::fused_moe_routing_kernel<
            kBlock, TD, opus::bf16_t, aiter::fmr::kFused, DB, NSHARED, DG>;
        (void)hipFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                  hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
        // The grid barrier deadlocks unless every block is co-resident.
        int max_blocks_per_cu = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_cu, reinterpret_cast<const void*>(kern), kBlock, shmem);
        TORCH_CHECK(max_blocks_per_cu >= 1,
                    "fused_moe_router_impl: kernel not resident (shmem=", shmem, ")");
        // Above the crossover, run the two halves as separate launches. Same
        // device code either way -- the split point IS the barrier -- but a
        // kernel boundary needs no co-residency, so each half gets a grid sized
        // to its own parallelism. The split trades the barrier for a second
        // launch's LDS allocation floor, so it only pays once the barrier's
        // grid-scaling overtakes that.
        //
        // AITER_MOE_ROUTING_SPLIT overrides the threshold rather than the
        // decision, so a sweep can retune the crossover without a rebuild.
        // 0 forces split at every M, a huge value forces fused. Read per call,
        // not cached: the tests flip it between launches in one process.
        int split_min = kSplitMinTokens;
        if(const char* ev = std::getenv("AITER_MOE_ROUTING_SPLIT"); ev && *ev)
        {
            char*      end = nullptr;
            const long v   = std::strtol(ev, &end, 10);
            TORCH_CHECK(end != ev && *end == '\0' && v >= 0 && v <= INT_MAX,
                        "AITER_MOE_ROUTING_SPLIT must be a non-negative token "
                        "count, got '", ev, "'");
            split_min = (int)v;
        }
        const bool split = M >= split_min;
        if(split)
        {
            auto* k1 = aiter::fmr::fused_moe_routing_kernel<
                kBlock, TD, opus::bf16_t, aiter::fmr::kPhase1, DB, NSHARED, DG>;
            auto* k23 = aiter::fmr::fused_moe_routing_kernel<
                kBlock, TD, opus::bf16_t, aiter::fmr::kPhase23, DB, NSHARED, DG>;
            (void)hipFuncSetAttribute(reinterpret_cast<const void*>(k1),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
            (void)hipFuncSetAttribute(reinterpret_cast<const void*>(k23),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
            const int G1  = std::max(1, std::min(M, prop.multiProcessorCount));
            const int G23 = GRID;
#define FMR_ARGS                                                                             \
    g, b, h, topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(),                       \
        sorted_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(),                        \
        sorted_expert_ids.data_ptr<int>(), num_valid_ids.data_ptr<int>(),                    \
        reinterpret_cast<opus::fp4_t*>(out_fp4.data_ptr()),                                  \
        reinterpret_cast<uint8_t*>(out_scale.data_ptr()),                                    \
        ws_tok_scale, moe_buf_ptr, moe_buf_elems, ws_sem, mask_ptr, M, E, topk,              \
        unit_size, group_size, cols, max_blocks, max_tokens, need_renorm,                    \
        (float)routed_scaling_factor, (float)shared_expert_weight, (int)ep_rank,             \
        (int)ep_size
            k1<<<G1, kBlock, shmem, stream>>>(FMR_ARGS);
            k23<<<G23, kBlock, shmem, stream>>>(FMR_ARGS);
#undef FMR_ARGS
            return; // returns from the lambda, not fused_moe_router_impl
        }
        // Only the fused path grid-barriers, so only it needs co-residency.
        TORCH_CHECK(GRID <= max_blocks_per_cu * prop.multiProcessorCount,
                    "fused_moe_router_impl: GRID ", GRID, " exceeds co-resident capacity");
        kern<<<GRID, kBlock, shmem, stream>>>(
            g, b, h, topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
            sorted_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(),
            sorted_expert_ids.data_ptr<int>(), num_valid_ids.data_ptr<int>(),
            reinterpret_cast<opus::fp4_t*>(out_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(out_scale.data_ptr()),
            ws_tok_scale, moe_buf_ptr, moe_buf_elems, ws_sem, mask_ptr,
            M, E, topk, unit_size, group_size, cols, max_blocks, max_tokens, need_renorm,
            (float)routed_scaling_factor, (float)shared_expert_weight, (int)ep_rank,
            (int)ep_size);
    };

    // NSHARED == 0 must reach the same instantiation as before this feature
    // existed, so the shared-expert lanes fold out entirely and the no-shared
    // config keeps its register count.
    // Selection is by shape alone -- the served dim decides the block, nothing consults a
    // model name, a path registry or an environment threshold.
    auto dispatch_gating = [&](auto bias_tag, auto nshared_tag, auto bs_tag) {
        if(gating_is_f32)
            launch_all(bias_tag, nshared_tag, bs_tag, float{});
        else
            launch_all(bias_tag, nshared_tag, bs_tag, opus::bf16_t{});
    };
    auto dispatch_block = [&](auto bias_tag, auto nshared_tag) {
        if(BlockSize == 256)
            dispatch_gating(bias_tag, nshared_tag, std::integral_constant<int, 256>{});
        else
            dispatch_gating(bias_tag, nshared_tag, std::integral_constant<int, 384>{});
    };
    auto dispatch_shared = [&](auto bias_tag) {
        if(n_shared == 0)
            dispatch_block(bias_tag, std::integral_constant<int, 0>{});
        else
            dispatch_block(bias_tag, std::integral_constant<int, 1>{});
    };

    if(bias_is_f32)
        dispatch_shared(float{});
    else
        dispatch_shared(opus::bf16_t{});
}

#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("fused_moe_router_impl", &fused_moe_router_impl,
          py::arg("gating"), py::arg("bias"), py::arg("hidden"),
          py::arg("topk_ids"), py::arg("topk_weights"), py::arg("sorted_ids"),
          py::arg("sorted_weights"), py::arg("sorted_expert_ids"),
          py::arg("num_valid_ids"), py::arg("out_fp4"), py::arg("out_scale"),
          py::arg("num_experts"), py::arg("topk"), py::arg("unit_size"),
          py::arg("group_size"), py::arg("need_renorm"),
          py::arg("routed_scaling_factor"), py::arg("workspace"),
          py::arg("expert_mask") = std::nullopt,
          py::arg("moe_buf") = std::nullopt,
          py::arg("num_fused_shared_experts") = 0,
          py::arg("shared_expert_weight") = 1.0,
          py::arg("ep_rank") = 0,
          py::arg("ep_size") = 1);
    m.def("fused_moe_router_workspace_size", &fused_moe_router_workspace_size,
          "workspace bytes for up to max_tokens tokens", py::arg("max_tokens"));
}
