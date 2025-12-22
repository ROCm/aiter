// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "kernels.h"

template <typename T, HFMemOp Op>
BenchmarkResult run_benchmark_return_result(const std::string& test_name, int num_cu, int64_t data_size_dwords)
{
    BenchmarkResult result;
    result.operation = test_name;
    
    if (std::is_same_v<T, float>) {
        result.vector_width = "32-bit";
    } else if (std::is_same_v<T, float2>) {
        result.vector_width = "64-bit";
    } else if (std::is_same_v<T, float4>) {
        result.vector_width = "128-bit";
    }
    
    result.size_mb = data_size_dwords * sizeof(float) / (1024.0 * 1024.0);
    
    const int grid_size = num_cu * OCCUPANCY_PER_CU;
    const int block_size = BLOCK_SIZE;

    const size_t num_elements_total = data_size_dwords / (sizeof(T) / sizeof(float));


    const size_t num_elements_per_block = (num_elements_total + grid_size - 1) / grid_size;
    const int iters = (num_elements_per_block + (block_size * UNROLL_FACTOR) - 1) / (block_size * UNROLL_FACTOR);
    
    const size_t num_elements_aligned = (size_t)grid_size * iters * block_size * UNROLL_FACTOR;
    const size_t data_size_bytes = num_elements_aligned * sizeof(T);


    T* d_data = nullptr;
    float* d_sum = nullptr;
    HIP_CHECK(hipMalloc(&d_data, data_size_bytes));
    HIP_CHECK(hipMalloc(&d_sum, sizeof(float)));
    HIP_CHECK(hipMemset(d_sum, 0, sizeof(float)));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    dim3 grid(grid_size, 1, 1);
    dim3 block(block_size, 1, 1);

    // warm up
    for(int i = 0; i < WARMUP; i++){
        if constexpr (Op == HFMemOp::GlobalLoad) {
            global_load_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
        } else if constexpr (Op == HFMemOp::GlobalLoadNT) {
            global_load_nt_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
        } else if constexpr (Op == HFMemOp::GlobalStore) {
            global_store_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::GlobalStoreNT) {
            global_store_nt_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::BufferStore) {
            buffer_store_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::BufferLoad) {  
            buffer_load_reg_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
         }else if constexpr (Op == HFMemOp::BufferLoadLDS) {
            buffer_load_lds_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::DsRead) {    
            size_t lds_size_bytes = block_size * UNROLL_FACTOR * sizeof(T);
            lds_load_kernel<T><<<grid, block, lds_size_bytes>>>(d_data,iters, d_sum);
        } else if constexpr (Op == HFMemOp::DsWrite) { 
            size_t lds_size_bytes = block_size * UNROLL_FACTOR * sizeof(T);
            lds_write_kernel<T><<<grid, block, lds_size_bytes>>>(d_data, iters);
        }
    }

    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < LOOP; ++i) {
        if constexpr (Op == HFMemOp::GlobalLoad) {
            global_load_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
        } else if constexpr (Op == HFMemOp::GlobalLoadNT) {
            global_load_nt_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
        } else if constexpr (Op == HFMemOp::GlobalStore) {
            global_store_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::GlobalStoreNT) {
            global_store_nt_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::BufferStore) {
            buffer_store_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::BufferLoad) {  
            buffer_load_reg_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters, d_sum);
         }else if constexpr (Op == HFMemOp::BufferLoadLDS) {
            buffer_load_lds_kernel<T><<<grid, block>>>(d_data, num_elements_per_block, iters);
        } else if constexpr (Op == HFMemOp::DsRead) {    
            lds_load_kernel<T><<<grid, block>>>(d_data, iters, d_sum);
        } else if constexpr (Op == HFMemOp::DsWrite) { 
            lds_write_kernel<T><<<grid, block>>>(d_data,  iters);
        }
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= LOOP;

    result.bandwidth_gb_s = (data_size_bytes / (1e9)) / (milliseconds / 1000.0);
    printf("%-40s: %.2f GB/s\n", test_name.c_str(), result.bandwidth_gb_s);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_sum));
    
    return result;
}


void write_results_to_file(const std::vector<BenchmarkResult>& results, const std::string& filename, 
                          const std::string& title, const hipDeviceProp_t& props, int num_cu) {
    FILE* md_file = fopen(filename.c_str(), "w");
    if (!md_file) {
        printf("Error: Failed to create markdown file %s.\n", filename.c_str());
        return;
    }

    fprintf(md_file, "# %s\n\n", title.c_str());
    fprintf(md_file, "## System Information\n\n");
    fprintf(md_file, "- **GPU**: %s\n", props.name);
    fprintf(md_file, "- **Compute Units**: %d\n", num_cu);
    fprintf(md_file, "- **OCCUPANCY_PER_CU**: %d\n", OCCUPANCY_PER_CU);
    fprintf(md_file, "- **Block Size**: %d\n", BLOCK_SIZE);
    fprintf(md_file, "- **Unroll Factor**: %d\n\n", UNROLL_FACTOR);
    
    fprintf(md_file, "\n\n## Results by Vector Width\n\n");
    
    std::vector<std::string> vector_widths = {"32-bit", "64-bit", "128-bit"};
    
    for (const auto& width : vector_widths) {
        fprintf(md_file, "### %s Operations\n\n", width.c_str());
        fprintf(md_file, "| Operation | Data Size (MB) | Bandwidth (GB/s) |\n");
        fprintf(md_file, "|-----------|---------------:|----------------:|\n");
        
        for (const auto& result : results) {
            if (result.vector_width == width) {
                    fprintf(md_file, "| %s | %.2f | %.2f |\n", 
                            result.operation.c_str(), result.size_mb, result.bandwidth_gb_s);
                
            }
        }
        fprintf(md_file, "\n");
    }
    
    fclose(md_file);
}

std::vector<BenchmarkResult> run_global_load_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Global Load (Cached) Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::GlobalLoad>(
            "global_load_dwordx2 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::GlobalLoad>(
            "global_load_dwordx4 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "global_load_results.md", 
                          "Global Load (Cached) Bandwidth Test Results", 
                          /*props=*/{}, num_cu);

    return results;
}

std::vector<BenchmarkResult> run_global_load_nt_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Global Load Non-Temporal Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::GlobalLoadNT>(
            "global_load_dwordx2_nt (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::GlobalLoadNT>(
            "global_load_dwordx4_nt (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "global_load_nt_results.md", 
                          "Global Load Non-Temporal Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_global_store_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Global Store Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::GlobalStore>(
            "global_store_dwordx2 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::GlobalStore>(
            "global_store_dwordx4 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "global_store_results.md", 
                          "Global Store Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_global_store_nt_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Global Store Non-Temporal Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::GlobalStoreNT>(
            "global_store_dwordx2_nt (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::GlobalStoreNT>(
            "global_store_dwordx4_nt (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "global_store_nt_results.md", 
                          "Global Store Non-Temporal Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_buffer_store_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Buffer Store Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::BufferStore>(
            "buffer_store_dwordx2 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::BufferStore>(
            "buffer_store_dwordx4 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "buffer_store_results.md", 
                          "Buffer Store Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_buffer_load_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Buffer Load Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::BufferLoad>(
            "buffer_load_dwordx2 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::BufferLoad>(
            "buffer_load_dwordx4 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "buffer_load_results.md", 
                          "Buffer Load Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_buffer_load_lds_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running Buffer Load LDS Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float, HFMemOp::BufferLoadLDS>(
            "buffer_load_dword (32-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "buffer_load_lds_results.md", 
                          "Buffer Load LDS Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}


std::vector<BenchmarkResult> run_lds_read_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running LDS Read Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::DsRead>(
            "ds_read_b64 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::DsRead>(
            "ds_read_b128 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "lds_Read_results.md", 
                          "LDS Read Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

std::vector<BenchmarkResult> run_lds_write_test(int num_cu, const std::vector<int64_t>& data_sizes) {
    std::vector<BenchmarkResult> results;
    printf("\n--- Running LDS Write Memory Bandwidth Test ---\n");
    
    for (int64_t dwords : data_sizes) {
        double size_mb = dwords * sizeof(float) / (1024.0 * 1024.0);
        printf("\n--- Testing with data size: %.2f MB ---\n", size_mb);

        results.push_back(run_benchmark_return_result<float2, HFMemOp::DsWrite>(
            "ds_write_b64 (64-bit)", num_cu, dwords));
        results.push_back(run_benchmark_return_result<float4, HFMemOp::DsWrite>(
            "ds_write_b128 (128-bit)", num_cu, dwords));
    }
    
    write_results_to_file(results, "lds_Write_results.md", 
                          "LDS Write Bandwidth Test Results", 
                          /*props=*/{}, num_cu);
    
    return results;
}

void run_all_tests(const std::string& test_name = "") {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    const int num_cu = props.multiProcessorCount;

    printf("=== GPU Memory Bandwidth Benchmark Suite ===\n");
    printf("GPU: %s, CUs: %d, BlockSize: %d, Unroll: %d, OCCUPANCY_PER_CU: %d\n\n",
           props.name, num_cu, BLOCK_SIZE, UNROLL_FACTOR, OCCUPANCY_PER_CU);

    std::vector<int64_t> data_sizes = {
    static_cast<int64_t>(64) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(256) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(320) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(512) * num_cu * BLOCK_SIZE
};

    std::vector<int64_t> lds_data_sizes = {
    static_cast<int64_t>(64) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(4096) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(10240) * num_cu * BLOCK_SIZE,
    static_cast<int64_t>(20480) * num_cu * BLOCK_SIZE
};

    if (test_name.empty() || test_name == "global_load") {
        run_global_load_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "global_load_nt") {
        run_global_load_nt_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "global_store") {
        run_global_store_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "global_store_nt") {
        run_global_store_nt_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "buffer_load") {
        run_buffer_load_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "buffer_store") {
        run_buffer_store_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "buffer_load_lds") {
        run_buffer_load_lds_test(num_cu, data_sizes);
    }
    
    if (test_name.empty() || test_name == "lds_read") {
        run_lds_read_test(num_cu, lds_data_sizes);
    }
    
    if (test_name.empty() || test_name == "lds_write") {
        run_lds_write_test(num_cu, lds_data_sizes);
    }

}

int main(int argc, char* argv[]) {
    
    std::vector<std::string> valid_tests = {
            "global_load", "global_load_nt", "global_store", "global_store_nt",
            "buffer_store", "buffer_load", "buffer_load_lds", "lds_read", "lds_write",
            ""
        };
    std::string test_name = "";
    run_all_tests(test_name);
    return 0;
}
