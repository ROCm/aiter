#include <sstream>
#include <optional>
#include <fmt/core.h>
#include <unistd.h>
#include "../utils.h"
#include "pa_decode_gluon_aot.h"
#include <fstream>

namespace aiter {

#define MD_NAME "transpose_query_gluon_kernel"

static inline int next_pow2(int n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

static inline std::string python_float_format(float x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8) << x;

    std::string s = oss.str();

    // remove trail zeros
    s.erase(s.find_last_not_of('0') + 1);

    // at least one digit after decimal point
    if (s.back() == '.')
        s.push_back('0');
    return s;
}

#define HIP_CHECK(cmd)                                \
    do                                                \
    {                                                 \
        hipError_t error = (cmd);                     \
        if(error != hipSuccess)                       \
        {                                             \
            fprintf(stderr,                           \
                    "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),         \
                    error,                            \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while(0)

void pa_decode_gluon_aot(torch::Tensor& output,
                         torch::Tensor& output_gluon,
                         torch::Tensor& query,
                         torch::Tensor& query_gluon,
                         torch::Tensor& query_scale_gluon,
                         torch::Tensor& key_cache,
                         torch::Tensor& value_cache,
                         torch::Tensor& context_lengths,
                         torch::Tensor& block_tables,
                         float softmax_scale,
                         int query_length,
                         int max_context_length,
                         int context_partition_size,
                         std::string compute_type,
                         torch::Tensor& query_scale,
                         torch::Tensor& key_scale,
                         torch::Tensor& value_scale,
                         torch::Tensor& exp_sums,
                         torch::Tensor& max_logits,
                         torch::Tensor& temporary_output,
                         std::optional<torch::Tensor> alibi_slopes) 
{
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int num_seqs    = output_gluon.size(0);
    int num_heads_total = output.size(1);
    int head_size   = output.size(2);

    int num_blocks  = key_cache.size(0);
    int num_kv_heads = key_cache.size(1);
    int key_head_split_dim = key_cache.size(2);
    int kv_block_size = key_cache.size(3);

    int query_group_size = num_heads_total / num_kv_heads;
    int equivalent_query_group_size =
        query_length * query_group_size;

    int is_causal = (query_length > 1);

    int max_context_partition_num =
        (max_context_length + context_partition_size - 1) / context_partition_size;

    int value_transposed = (value_cache.dim() == 5 ? 1 : 0);
    int query_quant_mode = -1;
    int kv_quant_mode = -1;

    int query_scale_stride_0 = 0;
    int kv_scale_stride_0 = 0;
    int kv_scale_stride_1 = 0;

    if (query_scale.defined()) {
        if (query_scale.numel() == 1) {
            query_quant_mode = 0;
        } else {
            query_quant_mode = 1;
            query_scale_stride_0 = query_scale.stride(0);
        }
    }

    if (key_scale.defined() && value_scale.defined()) {
        if (key_scale.numel() == 1)
            kv_quant_mode = 0;
        else {
            kv_quant_mode = 1;
            kv_scale_stride_0 = key_scale.stride(0);
            kv_scale_stride_1 = key_scale.stride(1);
        }
    }

    float fp8_max_value = 1.0f;
    int head_size_pow2 = next_pow2(head_size);
    int equi_query_group_size_pow2 =
        (equivalent_query_group_size < 16 ?
            16 :
            next_pow2(equivalent_query_group_size));

    std::list<std::string> args {
        compute_type,
        std::to_string(equi_query_group_size_pow2),
        std::to_string(head_size_pow2),
        std::to_string(kv_block_size),
        std::to_string(context_partition_size),
        std::to_string(query_quant_mode),
        std::to_string(kv_quant_mode),
        python_float_format(fp8_max_value),
        std::to_string(value_transposed),
        std::to_string(is_causal)
    };

    std::string func_name = get_default_func_name(MD_NAME, args);
    std::string folder = func_name;

    // std::ostringstream args_ss;
    // args_ss << "args = [";
    // for (auto it = args.begin(); it != args.end(); ++it) {
    //     args_ss << *it;
    //     if (std::next(it) != args.end()) args_ss << ", ";
    // }
    // args_ss << "]";
    // std::cout << args_ss.str() << std::endl;

    // std::cout << "func_name = " << func_name << std::endl;


    if (not_built(folder)) {
        std::string cmd = fmt::format(
            R"(python3 -m csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot \
                    --compute_type={compute_type} \
                    --equivalent_query_group_size={eq_group_size} \
                    --head_size={head_size} \
                    --kv_block_size={kv_block_size} \
                    --context_partition_size={context_partition_size} \
                    --query_quant_mode={query_quant_mode} \
                    --kv_quant_mode={kv_quant_mode} \
                    --fp8_max_value={fp8_max_value} \
                    --value_transposed={value_transposed} \
                    --is_causal={is_causal} \
                    --func_name={func_name})",
            fmt::arg("compute_type", compute_type),
            fmt::arg("eq_group_size", equivalent_query_group_size),
            fmt::arg("head_size", head_size),
            fmt::arg("kv_block_size", kv_block_size),
            fmt::arg("context_partition_size", context_partition_size),
            fmt::arg("query_quant_mode", query_quant_mode),
            fmt::arg("kv_quant_mode", kv_quant_mode),
            fmt::arg("fp8_max_value", fp8_max_value),
            fmt::arg("value_transposed", value_transposed),
            fmt::arg("is_causal", is_causal),
            fmt::arg("func_name", func_name)
        );
        execute_cmd(cmd);
    }


    int stride_output_seq  = output_gluon.stride(0); 
    int stride_output_head = output_gluon.stride(1);


    int stride_exp_sums_seq  = exp_sums.stride(0);
    int stride_exp_sums_head = exp_sums.stride(1);
    int stride_exp_sums_part = exp_sums.stride(2);

    int stride_logits_seq   = temporary_output.stride(0);
    int stride_logits_head  = temporary_output.stride(1);
    int stride_logits_part  = temporary_output.stride(2);
    int stride_logits_group = temporary_output.stride(3);

    int stride_query_seq  = query.stride(0);
    int stride_query_head = query.stride(1);

    int stride_key_block      = key_cache.stride(0);
    int stride_key_head       = key_cache.stride(1);
    int stride_key_head_split = key_cache.stride(2);
    int stride_key_block_elem = key_cache.stride(3);

    int stride_value_block      = value_cache.stride(0);
    int stride_value_head       = value_cache.stride(1);
    int stride_value_head_size  = value_cache.stride(2);

    int stride_block_table_seq = block_tables.stride(0);

    // std::cout << "func_name: " << func_name << std::endl;
    // std::cout << "folder: " << folder << std::endl;
    run_lib(func_name,
        folder,
        output_gluon.data_ptr(),
        exp_sums.data_ptr(),
        max_logits.data_ptr(),
        temporary_output.data_ptr(),
        query_gluon.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        block_tables.data_ptr(),
        context_lengths.data_ptr(),
        softmax_scale,
        (query_scale_gluon.defined() ? query_scale_gluon.data_ptr() : nullptr),
        (key_scale.defined() ? key_scale.data_ptr() : nullptr),
        (value_scale.defined() ? value_scale.data_ptr() : nullptr),
        output_gluon.stride(0),
        output_gluon.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        temporary_output.stride(0),
        temporary_output.stride(1),
        temporary_output.stride(2),
        temporary_output.stride(3),
        query_gluon.stride(0),
        query_gluon.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        query_scale_stride_0,
        kv_scale_stride_0,
        kv_scale_stride_1,
        num_seqs,
        num_kv_heads,
        max_context_partition_num,
        query_length,
        query_group_size,
        equivalent_query_group_size,
        head_size,
        reinterpret_cast<const void*>(stream)
);


}
#undef MD_NAME
} // namespace aiter

// torch::Tensor loadTensorFromFile(const std::string& fileName) {
//     // Open the file
//     std::ifstream fin(fileName, std::ios::in | std::ios::binary);
//     if (!fin) {
//         throw std::runtime_error("Failed to open file: " + fileName);
//     }

//     // Get the file size
//     fin.seekg(0, std::ios::end);
//     size_t fileSize = fin.tellg();
//     fin.seekg(0, std::ios::beg);

//     // Read the file content into a vector
//     std::vector<char> pickledData(fileSize);
//     fin.read(pickledData.data(), fileSize);
//     fin.close();

//     // Deserialize the Tensor
//     return torch::pickle_load(pickledData).toTensor();
// }

// int main() {
//     // pid_t pid = getpid();

//     // std::cout << "====================================================\n";
//     // std::cout << " Process started. You can attach gdb using:\n\n";
//     // std::cout << "    pid " << pid << "\n\n";
//     // std::cout << "====================================================\n";
//     // std::this_thread::sleep_for(std::chrono::seconds(15));
//     torch::Tensor query_scale, query_scale_gluon;
//     auto output_gluon = torch::empty({1, 160, 128}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::Device(torch::kCUDA)));
//     auto query_gluon = torch::empty({1, 160, 128}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::Device(torch::kCUDA)));

//     auto output = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/out.pt").to(torch::kCUDA);
//     auto query = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/query.pt").to(torch::kCUDA);
//     auto key_cache = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/key_cache.pt").view(torch::kFloat8_e4m3fnuz).to(torch::kCUDA);
//     auto value_cache = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/value_cache.pt").view(torch::kFloat8_e4m3fnuz).to(torch::kCUDA);
//     auto context_lengths = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/seq_lens.pt").to(torch::kCUDA);
//     auto block_tables = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/block_tables.pt").to(torch::kCUDA);
//     auto key_scale = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/k_scale.pt").to(torch::kCUDA);
//     auto value_scale = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/v_scale.pt").to(torch::kCUDA);
//     auto exp_sums = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/exp_sums.pt").to(torch::kCUDA);
//     auto max_logits = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/max_logits.pt").to(torch::kCUDA);
//     auto temporary_output = loadTensorFromFile("/mnt/raid0/yilin/repo/aiter/zzz2/tmp_out.pt").to(torch::kCUDA);

//     float scale = 0.09;
//     int64_t query_length = 4;
//     int64_t max_context_length = 4;
//     int64_t context_partition_size = 256;
//     aiter::pa_decode_gluon_aot(
//         output,
//         output_gluon,
//         query,
//         query_gluon,
//         query_scale_gluon,
//         key_cache,
//         value_cache,
//         context_lengths,
//         block_tables,
//         scale,
//         query_length,
//         max_context_length,
//         context_partition_size,
//         std::string("fp8e4b8"),
//         query_scale,
//         key_scale,
//         value_scale,
//         exp_sums,
//         max_logits,
//         temporary_output,
//         std::nullopt
//     );

//     return 0;
// }
