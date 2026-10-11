#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdlib>

static int gdn_sort_threshold() {
    static int threshold = []() {
        const char* env = std::getenv("HIP_GDN_SORT_IDX_BS");
        return env ? std::atoi(env) : 192;
    }();
    return threshold;
}

// Cache sorted indices and permutation across layers in one decode step.
static struct {
    const void* last_ptr = nullptr;
    int last_bs = 0;
    torch::Tensor sorted_indices;
    torch::Tensor perm_i32;
} sort_cache;

extern "C" {
void launch_gdn_decode_iasm(
    const void* query,
    const void* key,
    const void* value,
    const void* a_input,
    const void* b_input,
    const void* dt_bias,
    const void* A_log,
    const void* indices,
    void* state,
    void* output,
    int batch_size,
    int seq_length,
    int num_v_blocks,
    bool use_qk_l2norm,
    float scale,
    int num_k_heads,
    int num_v_heads,
    const int* batch_perm,
    hipStream_t stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "hip_gdn_decode_asm_inplace",
        [](torch::Tensor query,
           torch::Tensor key,
           torch::Tensor value,
           torch::Tensor a,
           torch::Tensor b,
           torch::Tensor dt_bias,
           torch::Tensor A_log,
           torch::Tensor indices,
           torch::Tensor state,
           torch::Tensor output,
           int batch_size,
           int seq_length,
           int num_v_blocks,
           bool use_qk_l2norm,
           float scale,
           int num_k_heads,
           int num_v_heads) {
            auto stream = at::hip::getCurrentHIPStream().stream();
            const int* batch_perm_ptr = nullptr;
            torch::Tensor sorted_indices;
            torch::Tensor perm_i32;
            const int sort_threshold = gdn_sort_threshold();

            if (sort_threshold > 0 && batch_size >= sort_threshold &&
                num_k_heads == 2 && num_v_heads == 8) {
                const void* cur_ptr = indices.data_ptr();
                if (sort_cache.last_ptr == cur_ptr &&
                    sort_cache.last_bs == batch_size) {
                    sorted_indices = sort_cache.sorted_indices;
                    perm_i32 = sort_cache.perm_i32;
                } else {
                    auto perm_i64 = torch::argsort(
                        indices, /*dim=*/int64_t(0), /*descending=*/false);
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
                batch_size, seq_length, num_v_blocks, use_qk_l2norm, scale,
                num_k_heads, num_v_heads, batch_perm_ptr, stream);

            auto err = hipGetLastError();
            TORCH_CHECK(
                err == hipSuccess,
                "hip_gdn_decode_asm_inplace launch failed: ", hipGetErrorString(err),
                " (BS=", batch_size, " SQ=", seq_length,
                " NKH=", num_k_heads, " NVH=", num_v_heads,
                " NVB=", num_v_blocks, " l2norm=", use_qk_l2norm,
                " dt_dtype=", dt_bias.dtype(), " q_dtype=", query.dtype(),
                " state_dtype=", state.dtype(), ")");
        },
        "GDN decode ASM kernel (inline asm, state [pool, HV, V, K])");
}
