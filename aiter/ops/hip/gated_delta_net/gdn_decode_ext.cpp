#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdlib>

static int gdn_sort_threshold() {
    // BS=128 with sorted path under-saturates HBM (~1.62 TB/s -> 79us).
    // The unsorted path at BS=128 hits ~2.16 TB/s -> 59us (-25% kernel time, +30% vs FlyDSL).
    // Sort still helps at BS=256 (97us vs 103us). Setting threshold=192 keeps sort only for
    // BS in [192, +inf), so BS=128 uses the faster unsorted kernel and BS=256 keeps sort.
    static int t = []() {
        const char* e = std::getenv("HIP_GDN_SORT_IDX_BS");
        return e ? std::atoi(e) : 192;
    }();
    return t;
}

// Cache sorted indices + permutation across layers in a decode step.
// All GDN layers in one step share the same indices tensor, so sorting
// once and reusing is a ~64x reduction in sort overhead.
static struct {
    const void* last_ptr = nullptr;
    int last_bs = 0;
    torch::Tensor sorted_indices;
    torch::Tensor perm_i32;
} sort_cache;

extern "C" {
void launch_gdn_decode_iasm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    const int* batch_perm,
    hipStream_t stream
);

void launch_gdn_decode_iasm_generic(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_tuned(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    hipStream_t stream
);

void launch_gdn_decode_tuned_kv(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_tuned_vk(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_state_transpose(
    void* state, const void* indices,
    int batch_size, int num_v_heads,
    hipStream_t stream
);

void launch_state_transpose_multi_layer(
    void* state_base,
    const void* indices,
    const void* slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats,
    hipStream_t stream
);

void launch_gdn_decode_kv4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_fused(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);
}

void hip_gdn_decode_tuned_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        stream
    );
}

void hip_gdn_decode_tuned_kv_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned_kv(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        num_k_heads, num_v_heads,
        stream
    );
}

void hip_gdn_decode_tuned_vk_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned_vk(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        num_k_heads, num_v_heads,
        stream
    );
}

void hip_state_transpose(
    torch::Tensor state, torch::Tensor indices,
    int batch_size, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_state_transpose(
        state.data_ptr(), indices.data_ptr(),
        batch_size, num_v_heads, stream);
}

void hip_state_transpose_multi_layer(
    torch::Tensor state_base,
    torch::Tensor indices,
    torch::Tensor slot_layout,
    int64_t target_layout,
    int64_t num_layers,
    int64_t batch_size,
    int64_t num_v_heads,
    int64_t layer_stride_floats
) {
    TORCH_CHECK(slot_layout.scalar_type() == torch::kInt8,
        "slot_layout must be int8, got ", slot_layout.scalar_type());
    TORCH_CHECK(state_base.is_contiguous(),
        "state_base must be contiguous");
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_state_transpose_multi_layer(
        state_base.data_ptr(),
        indices.data_ptr(),
        slot_layout.data_ptr(),
        (int)target_layout,
        (int)num_layers,
        (int)batch_size,
        (int)num_v_heads,
        layer_stride_floats,
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hip_gdn_decode_tuned_inplace", &hip_gdn_decode_tuned_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, V, K])");
    m.def("hip_gdn_decode_tuned_kv_inplace", &hip_gdn_decode_tuned_kv_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, K, V] - sglang compatible)");
    m.def("hip_gdn_decode_tuned_vk_inplace", &hip_gdn_decode_tuned_vk_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, V, K] with runtime heads)");
    m.def("hip_state_transpose", &hip_state_transpose,
          "In-place 128x128 state transpose [K,V] <-> [V,K]");
    m.def("hip_state_transpose_multi_layer", &hip_state_transpose_multi_layer,
          "Batched multi-layer in-place 128x128 transpose with per-slot layout gating "
          "(skips slots already in target layout)");
    m.def("hip_gdn_decode_vk_auto_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_state_transpose(state.data_ptr(), indices.data_ptr(),
                               batch_size, num_v_heads, stream);
        launch_gdn_decode_tuned_vk(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
        launch_state_transpose(state.data_ptr(), indices.data_ptr(),
                               batch_size, num_v_heads, stream);
    }, "GDN decode VK kernel with auto state transpose (single C++ call)");
    m.def("hip_gdn_decode_fused_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_fused(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode FUSED kernel (LDS transpose, [K,V] state, single kernel)");
    m.def("hip_gdn_decode_asm_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        auto err_before = hipGetLastError();

        const int* batch_perm_ptr = nullptr;
        torch::Tensor sorted_indices, perm_i32;
        int sort_thr = gdn_sort_threshold();
        if (sort_thr > 0 && batch_size >= sort_thr &&
            num_k_heads == 2 && num_v_heads == 8) {
            const void* cur_ptr = indices.data_ptr();
            if (sort_cache.last_ptr == cur_ptr &&
                sort_cache.last_bs == batch_size) {
                sorted_indices = sort_cache.sorted_indices;
                perm_i32 = sort_cache.perm_i32;
            } else {
                auto perm_i64 = torch::argsort(indices, /*dim=*/int64_t(0), /*descending=*/false);
                sorted_indices = indices.index_select(0, perm_i64);
                perm_i32 = perm_i64.to(torch::kInt32);
                sort_cache.last_ptr = cur_ptr;
                sort_cache.last_bs = batch_size;
                sort_cache.sorted_indices = sorted_indices;
                sort_cache.perm_i32 = perm_i32;
            }
            batch_perm_ptr = perm_i32.data_ptr<int>();
        }

        launch_gdn_decode_iasm(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(),
            batch_perm_ptr ? sorted_indices.data_ptr() : indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads,
            batch_perm_ptr, stream);
        auto err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess,
            "hip_gdn_decode_asm_inplace launch failed: ", hipGetErrorString(err),
            " (BS=", batch_size, " SQ=", seq_length, " NKH=", num_k_heads,
            " NVH=", num_v_heads, " NVB=", num_v_blocks,
            " l2norm=", use_qk_l2norm, " dt_dtype=", dt_bias.dtype(),
            " q_dtype=", query.dtype(), " state_dtype=", state.dtype(), ")");
    }, "GDN decode ASM kernel (inline asm reduces, state [V,K], template-specialized heads)");
    m.def("hip_gdn_decode_asm_generic_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_generic(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode ASM generic fallback kernel (pre-extreme dispatch)");
    m.def("hip_gdn_decode_kv4_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_kv4(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode KV4 kernel (state layout: [pool, HV, K, V] with float4 along V)");

}
