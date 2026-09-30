// Dedicated Q256 producer ABI; independent of AITER asm_fmoe KernelArgs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace q256 {
struct alignas(16) ProducerArgs {
    void* output;
    std::uint64_t reserved_after_output;
    void* activations;
    std::uint64_t reserved_after_activations;
    void* gate_up_weights;
    std::uint64_t reserved_after_gate_up_weights;
    void* persistent_counts;
    std::uint64_t reserved_after_persistent_counts;
    void* down_weights;
    std::uint64_t reserved_after_down_weights;
    void* activation_scales;
    std::uint64_t reserved_after_activation_scales;
    void* gate_up_scales;
    std::uint64_t reserved_after_gate_up_scales;
    void* down_scales;
    std::uint64_t reserved_after_down_scales;
    void* smoothing_scales;
    std::uint64_t reserved_after_smoothing_scales;
    void* sorted_token_ids;
    std::uint64_t reserved_after_sorted_token_ids;
    void* routing_weights;
    std::uint64_t reserved_after_routing_weights;
    void* sorted_expert_ids;
    std::uint64_t reserved_after_sorted_expert_ids;
    std::uint32_t model_width;
    std::uint32_t reserved_after_model_width[3];
    std::uint32_t intermediate_width;
    std::uint32_t reserved_after_intermediate_width[3];
    std::uint32_t token_count;
    std::uint32_t reserved_after_token_count[3];
    std::uint32_t expert_count;
    std::uint32_t reserved_after_expert_count[3];
    std::uint32_t activation_row_stride;
    std::uint32_t reserved_after_activation_row_stride[3];
    std::uint32_t projection_row_stride;
    std::uint32_t reserved_after_projection_row_stride[3];
    std::uint32_t down_row_stride;
    std::uint32_t reserved_after_down_row_stride[3];
    std::uint32_t output_row_stride;
    std::uint32_t reserved_after_output_row_stride[3];
    std::uint32_t projection_expert_stride;
    std::uint32_t reserved_after_projection_expert_stride[3];
    std::uint32_t down_expert_stride;
    std::uint32_t reserved_after_down_expert_stride[3];
    std::uint32_t projection_scale_expert_stride;
    std::uint32_t reserved_after_projection_scale_expert_stride[3];
    std::uint32_t down_scale_expert_stride;
    std::uint32_t reserved_after_down_scale_expert_stride[3];
    std::uint32_t smoothing_scale_expert_stride;
    std::uint32_t reserved_after_smoothing_scale_expert_stride[3];
    std::uint32_t top_k;
    std::uint32_t reserved_after_top_k[3];
    std::uint32_t total_workgroups;
    std::uint32_t reserved_after_total_workgroups[3];
    std::uint32_t hidden_tiles;
    std::uint32_t reserved_after_hidden_tiles[3];
};

static_assert(sizeof(void*) == 8, "Q256 requires 64-bit pointers");
static_assert(std::is_standard_layout_v<ProducerArgs>);
static_assert(std::is_trivially_copyable_v<ProducerArgs>);
static_assert(alignof(ProducerArgs) == 16);
static_assert(sizeof(ProducerArgs) == 448);
static_assert(offsetof(ProducerArgs, output) == 0);
static_assert(offsetof(ProducerArgs, activations) == 16);
static_assert(offsetof(ProducerArgs, gate_up_weights) == 32);
static_assert(offsetof(ProducerArgs, persistent_counts) == 48);
static_assert(offsetof(ProducerArgs, down_weights) == 64);
static_assert(offsetof(ProducerArgs, activation_scales) == 80);
static_assert(offsetof(ProducerArgs, gate_up_scales) == 96);
static_assert(offsetof(ProducerArgs, down_scales) == 112);
static_assert(offsetof(ProducerArgs, smoothing_scales) == 128);
static_assert(offsetof(ProducerArgs, sorted_token_ids) == 144);
static_assert(offsetof(ProducerArgs, routing_weights) == 160);
static_assert(offsetof(ProducerArgs, sorted_expert_ids) == 176);
static_assert(offsetof(ProducerArgs, model_width) == 192);
static_assert(offsetof(ProducerArgs, intermediate_width) == 208);
static_assert(offsetof(ProducerArgs, token_count) == 224);
static_assert(offsetof(ProducerArgs, expert_count) == 240);
static_assert(offsetof(ProducerArgs, activation_row_stride) == 256);
static_assert(offsetof(ProducerArgs, projection_row_stride) == 272);
static_assert(offsetof(ProducerArgs, down_row_stride) == 288);
static_assert(offsetof(ProducerArgs, output_row_stride) == 304);
static_assert(offsetof(ProducerArgs, projection_expert_stride) == 320);
static_assert(offsetof(ProducerArgs, down_expert_stride) == 336);
static_assert(offsetof(ProducerArgs, projection_scale_expert_stride) == 352);
static_assert(offsetof(ProducerArgs, down_scale_expert_stride) == 368);
static_assert(offsetof(ProducerArgs, smoothing_scale_expert_stride) == 384);
static_assert(offsetof(ProducerArgs, top_k) == 400);
static_assert(offsetof(ProducerArgs, total_workgroups) == 416);
static_assert(offsetof(ProducerArgs, hidden_tiles) == 432);
}  // namespace q256
