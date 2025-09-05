#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" hipError_t hip_ssytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    float* dA,
    int lda,
    float* dD,
    float* dE,
    float* dTau);

extern "C" hipError_t hip_dsytrd(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    double* dA,
    int lda,
    double* dD,
    double* dE,
    double* dTau);

// Generate a random symmetric matrix (column-major)
template<typename scalar_t>
void generate_symmetric_matrix(std::vector<scalar_t>& A, int n, unsigned int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar_t> dist(static_cast<scalar_t>(0), static_cast<scalar_t>(1));
    std::fill(A.begin(), A.end(), static_cast<scalar_t>(0));
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            scalar_t value = dist(rng);
            A[i + j * n] = value;
            if (i != j) {
                A[j + i * n] = value;
            }
        }
    }
}

// Struct to hold benchmark results
template<typename scalar_t>
struct BenchmarkResult {
    double hip_avg_time_ms = 0.0;
    double rocsolver_avg_time_ms = 0.0;
    double speedup_rocsolver_over_hip = 0.0;
    bool   validated = false;
    scalar_t max_abs_diff_D = static_cast<scalar_t>(0);
    scalar_t max_abs_diff_E = static_cast<scalar_t>(0);
    scalar_t max_abs_diff_Tau = static_cast<scalar_t>(0);
};

// Benchmark custom and rocSOLVER tridiagonalization, optionally validate results
template<typename scalar_t>
BenchmarkResult<scalar_t> benchmark(int n, bool validate, int iterations, int warmup) {
    const int lda = n;
    BenchmarkResult<scalar_t> result;

    std::vector<scalar_t> hA(n * n);
    std::vector<scalar_t> hD_hip(n), hE_hip(std::max(0, n - 1)), hTau_hip(std::max(0, n - 1));
    std::vector<scalar_t> hD_ref(n), hE_ref(std::max(0, n - 1)), hTau_ref(std::max(0, n - 1));
    scalar_t *dA_hip, *dD_hip, *dE_hip, *dTau_hip;
    hipMalloc(&dA_hip, n * lda * sizeof(scalar_t));
    hipMalloc(&dD_hip, n * sizeof(scalar_t));
    hipMalloc(&dE_hip, (n - 1) * sizeof(scalar_t));
    hipMalloc(&dTau_hip, (n - 1) * sizeof(scalar_t));
    scalar_t *dA_ref, *dD_ref, *dE_ref, *dTau_ref;
    hipMalloc(&dA_ref, n * lda * sizeof(scalar_t));
    hipMalloc(&dD_ref, n * sizeof(scalar_t));
    hipMalloc(&dE_ref, (n - 1) * sizeof(scalar_t));
    hipMalloc(&dTau_ref, (n - 1) * sizeof(scalar_t));

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    generate_symmetric_matrix<scalar_t>(hA, n);

    for (int w = 0; w < warmup; ++w) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        
        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        }
    }
    double hip_total_us = 0.0;
    double ref_total_us = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        auto start_hip = std::chrono::high_resolution_clock::now();
        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
        }
        auto end_hip = std::chrono::high_resolution_clock::now();
        hip_total_us += std::chrono::duration<double, std::micro>(end_hip - start_hip).count();
        auto start_ref = std::chrono::high_resolution_clock::now();
        if constexpr (std::is_same_v<scalar_t, float>) {
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        }
        auto end_ref = std::chrono::high_resolution_clock::now();
        ref_total_us += std::chrono::duration<double, std::micro>(end_ref - start_ref).count();
    }

    result.hip_avg_time_ms = (hip_total_us / iterations) / 1000.0;
    result.rocsolver_avg_time_ms = (ref_total_us / iterations) / 1000.0;
    if (result.hip_avg_time_ms > 0.0)
        result.speedup_rocsolver_over_hip = result.rocsolver_avg_time_ms / result.hip_avg_time_ms;

    if (validate) {
        generate_symmetric_matrix<scalar_t>(hA, n, 12345u);
        hipMemcpy(dA_hip, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);
        hipMemcpy(dA_ref, hA.data(), n * lda * sizeof(scalar_t), hipMemcpyHostToDevice);

        if constexpr (std::is_same_v<scalar_t, float>) {
            hip_ssytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_ssytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        } else {
            hip_dsytrd(handle, rocblas_fill_lower, n, dA_hip, lda, dD_hip, dE_hip, dTau_hip);
            rocsolver_dsytrd(handle, rocblas_fill_lower, n, dA_ref, lda, dD_ref, dE_ref, dTau_ref);
        }

        hipMemcpy(hD_hip.data(), dD_hip, n * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hE_hip.data(), dE_hip, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_hip.data(), dTau_hip, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hD_ref.data(), dD_ref, n * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hE_ref.data(), dE_ref, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);
        hipMemcpy(hTau_ref.data(), dTau_ref, (n - 1) * sizeof(scalar_t), hipMemcpyDeviceToHost);

        scalar_t max_d = static_cast<scalar_t>(0), max_e = static_cast<scalar_t>(0), max_tau = static_cast<scalar_t>(0);
        for (int i = 0; i < n; ++i) {
            max_d = std::max(max_d, std::abs(hD_hip[i] - hD_ref[i]));
        }
        for (int i = 0; i < n - 1; ++i) {
            max_e = std::max(max_e, std::abs(hE_hip[i] - hE_ref[i]));
            max_tau = std::max(max_tau, std::abs(hTau_hip[i] - hTau_ref[i]));
        }
        result.max_abs_diff_D = max_d;
        result.max_abs_diff_E = max_e;
        result.max_abs_diff_Tau = max_tau;
        result.validated = true;
    }

    hipFree(dA_hip); hipFree(dD_hip); hipFree(dE_hip); hipFree(dTau_hip);
    hipFree(dA_ref); hipFree(dD_ref); hipFree(dE_ref); hipFree(dTau_ref);
    rocblas_destroy_handle(handle);

    return result;
}

// Print table header for benchmark results
void print_table_header(const std::string& precision, bool validate) {
    printf("Matrix Size | %s (ms) | rocSOLVER (ms) | Speedup",
           (precision == "float") ? "hip_ssytrd" : "hip_dsytrd");

    if (validate) {
        printf(" |   max|ΔD|   |   max|ΔE|   |   max|ΔTau| \n");
        printf("--------------------------------------------------------------------------------------------------\n");
    } else {
        printf("\n");
        printf("--------------------------------------------------------\n");
    }
}

// Print a single row of benchmark results
template<typename scalar_t>
void print_comparison_result(int n, const BenchmarkResult<scalar_t>& result, bool validate) {
    printf("%11d | %15.3f | %14.3f | %7.3f", n,
           result.hip_avg_time_ms,
           result.rocsolver_avg_time_ms,
           result.speedup_rocsolver_over_hip);
    if (validate && result.validated) {
        printf(" | %11.3e | %11.3e | %11.3e",
               result.max_abs_diff_D,
               result.max_abs_diff_E,
               result.max_abs_diff_Tau);
    }
    printf("\n");
}

// Print usage/help message
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -n size1 size2 ...    Matrix sizes to test\n");
    printf("  -p precision          Precision: 'float' or 'double' (default: float)\n");
    printf("  -v                    Enable validation\n");
    printf("  -i iterations         Number of benchmark iterations (default: 10)\n");
    printf("  -w warmup_runs        Number of warmup runs (default: 3)\n");
    printf("  -h                    Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -n 128 256 -p float -v\n", program_name);
}

// Main entry point: parse arguments, run benchmarks, print results
int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes;
    bool validate = false;
    int iterations = 10;
    int warmup = 3;
    std::string precision = "float";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                matrix_sizes.push_back(atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            precision = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (matrix_sizes.empty()) {
        printf("Error: No matrix sizes specified.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (precision != "float" && precision != "double") {
        printf("Error: Invalid precision '%s'. Use 'float' or 'double'.\n\n", precision.c_str());
        print_usage(argv[0]);
        return 1;
    }

    printf("Householder Tridiagonalization Benchmark\n");
    printf("=========================================\n");
    printf("Precision: %s\n", precision.c_str());
    printf("Iterations: %d\n", iterations);
    printf("Warmup runs: %d\n", warmup);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    
    print_table_header(precision, validate);
    
    if (precision == "float") {
        for (int n : matrix_sizes) {
            BenchmarkResult<float> result = benchmark<float>(n, validate, iterations, warmup);
            print_comparison_result(n, result, validate);
        }
    } else {
        for (int n : matrix_sizes) {
            BenchmarkResult<double> result = benchmark<double>(n, validate, iterations, warmup);
            print_comparison_result(n, result, validate);
        }
    }

    printf("\n");

    return 0;
}