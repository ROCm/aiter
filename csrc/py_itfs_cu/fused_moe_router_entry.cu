// Host entry for the fused MoE routing kernel (module_fused_moe_router).

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>
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
    std::optional<torch::Tensor> moe_buf)
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
    constexpr int BlockSize = 256;
    constexpr int TD        = 16; // cols / TD must equal BlockSize
    TORCH_CHECK(cols == BlockSize * TD, "fused_moe_router_impl: cols must be ", BlockSize * TD);
    TORCH_CHECK(E <= 2 * BlockSize,
                "fused_moe_router_impl: num_experts must be <= ", 2 * BlockSize);
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
    // Phase 1 selects within one wave, so a larger topk would silently drop
    // every winner past lane 63.
    TORCH_CHECK(topk > 0 && topk <= 64,
                "fused_moe_router_impl: topk must be in 1..64 (phase 1 selects "
                "within one wave), got ", topk);
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

    // The mask is indexed by global expert id. vLLM sizes it longer than E
    // (expert_map_manager.py) with trailing sentinel and fused-shared slots;
    // neither is a routed expert, so only the first E are read. Local ids still
    // match the stock path because local(e) is an *exclusive* prefix.
    const int* mask_ptr = nullptr;
    if(expert_mask.has_value() && expert_mask->defined())
    {
        TORCH_CHECK(expert_mask->numel() >= E,
                    "fused_moe_router_impl: expert_mask must have at least num_experts (",
                    E, ") entries, got ", expert_mask->numel());
        TORCH_CHECK(expert_mask->scalar_type() == torch::kInt32,
                    "fused_moe_router_impl: expert_mask must be int32");
        TORCH_CHECK(expert_mask->is_contiguous(),
                    "fused_moe_router_impl: expert_mask must be contiguous");
        mask_ptr = expert_mask->data_ptr<int>();
    }

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Grid width multiplies barrier cost, so size it to the work and keep it
    // under num_cu so every block stays resident. Phase 1 wants one block per
    // token; phase 3's rows do not shrink with M, so small M still needs width
    // -- hence the floor.
    int GRID = std::max(M, 16);
    int dev_id = 0;
    hipGetDevice(&dev_id);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev_id);
    GRID = std::max(1, std::min(GRID, prop.multiProcessorCount));

    const int total_routed_rows = M * topk;
    const int max_blocks        = (int)sorted_expert_ids.numel();
    const int max_tokens        = (int)sorted_ids.numel();
    const int scaleN_valid      = (cols + group_size - 1) / group_size;
    const int scaleN_pad =
        ((scaleN_valid + kScalesPerThread - 1) / kScalesPerThread) * kScalesPerThread;

    const size_t shmem =
        aiter::fmr::LdsLayout(BlockSize, total_routed_rows, E, mask_ptr != nullptr).bytes;

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
    check_bf16(gating, "gating");
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

    const opus::bf16_t* g = reinterpret_cast<const opus::bf16_t*>(gating.data_ptr());
    const opus::bf16_t* h = reinterpret_cast<const opus::bf16_t*>(hidden.data_ptr());

    // The bias is not necessarily bf16 and, unlike the stock
    // biased_grouped_topk wrapper, this path does not coerce it -- reading fp32
    // as bf16 silently mis-routes. Dispatch on the real dtype rather than
    // converting on the host: the bias is only read in phase 1's top-k.
    TORCH_CHECK(bias.scalar_type() == at::kFloat ||
                    bias.scalar_type() == at::kBFloat16,
                "fused_moe_router_impl: correction bias must be float32 or "
                "bfloat16, got ", bias.scalar_type());
    const bool bias_is_f32 = bias.scalar_type() == at::kFloat;

    // One body, instantiated per bias dtype.
    auto launch_all = [&](auto bias_tag) {
        using DB = decltype(bias_tag);
        const DB* b = reinterpret_cast<const DB*>(bias.data_ptr());
        auto* kern = aiter::fmr::fused_moe_routing_kernel<
            BlockSize, TD, opus::bf16_t, aiter::fmr::kFused, DB>;
        (void)hipFuncSetAttribute(reinterpret_cast<const void*>(kern),
                                  hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
        // The grid barrier deadlocks unless every block is co-resident.
        int max_blocks_per_cu = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_cu, reinterpret_cast<const void*>(kern), BlockSize, shmem);
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
                BlockSize, TD, opus::bf16_t, aiter::fmr::kPhase1, DB>;
            auto* k23 = aiter::fmr::fused_moe_routing_kernel<
                BlockSize, TD, opus::bf16_t, aiter::fmr::kPhase23, DB>;
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
        (float)routed_scaling_factor
            k1<<<G1, BlockSize, shmem, stream>>>(FMR_ARGS);
            k23<<<G23, BlockSize, shmem, stream>>>(FMR_ARGS);
#undef FMR_ARGS
            return; // returns from the lambda, not fused_moe_router_impl
        }
        // Only the fused path grid-barriers, so only it needs co-residency.
        TORCH_CHECK(GRID <= max_blocks_per_cu * prop.multiProcessorCount,
                    "fused_moe_router_impl: GRID ", GRID, " exceeds co-resident capacity");
        kern<<<GRID, BlockSize, shmem, stream>>>(
            g, b, h, topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(),
            sorted_ids.data_ptr<int>(), sorted_weights.data_ptr<float>(),
            sorted_expert_ids.data_ptr<int>(), num_valid_ids.data_ptr<int>(),
            reinterpret_cast<opus::fp4_t*>(out_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(out_scale.data_ptr()),
            ws_tok_scale, moe_buf_ptr, moe_buf_elems, ws_sem, mask_ptr,
            M, E, topk, unit_size, group_size, cols, max_blocks, max_tokens, need_renorm,
            (float)routed_scaling_factor);
    };

    if(bias_is_f32)
        launch_all(float{});
    else
        launch_all(opus::bf16_t{});
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
          py::arg("moe_buf") = std::nullopt);
    m.def("fused_moe_router_workspace_size", &fused_moe_router_workspace_size,
          "workspace bytes for up to max_tokens tokens", py::arg("max_tokens"));
}
