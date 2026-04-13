// test_topk_v3.cu — Standalone benchmark & correctness test for V3 multi-block radix topk
//
// Compile:
//   hipcc -O3 -std=c++17 test_topk_v3.cu -o test_topk_v3 --offload-arch=gfx942
//
// Run:
//   ./test_topk_v3 [LEN] [K] [BATCH]
//   defaults: LEN=60000  K=2048  BATCH=1

#define TOPK_STANDALONE_TEST
#include "topk_bothPD_mulblocks_v3.cu"

int main(int argc, char** argv)
{
    const int LEN   = (argc > 1) ? std::atoi(argv[1]) : 60000;
    const int K     = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int BATCH = (argc > 3) ? std::atoi(argv[3]) : 1;

    constexpr int  WARMUP     = 50;
    constexpr int  REPEAT     = 200;
    constexpr bool IS_LARGEST = true;
    constexpr int  NEXT_N     = 1;

    printf("=== V3 Radix TopK Test ===\n");
    printf("LEN=%d  K=%d  BATCH=%d  WARMUP=%d  REPEAT=%d\n\n", LEN, K, BATCH, WARMUP, REPEAT);

    // ---- Host data ----
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1e6f, 1e6f);

    std::vector<float> h_in(BATCH * LEN);
    for(auto& v : h_in) { v = dist(rng); }

    std::vector<int> h_rowStarts(BATCH, 0);
    std::vector<int> h_rowEnds(BATCH, LEN);
    std::vector<int> h_seqLens(BATCH, LEN);

    // ---- Device allocations ----
    float* d_in        = nullptr;
    int*   d_out_idx   = nullptr;
    int*   d_rowStarts = nullptr;
    int*   d_rowEnds   = nullptr;
    int*   d_seqLens   = nullptr;

    HIP_CALL(hipMalloc(&d_in,        sizeof(float) * BATCH * LEN));
    HIP_CALL(hipMalloc(&d_out_idx,   sizeof(int)   * BATCH * K));
    HIP_CALL(hipMalloc(&d_rowStarts, sizeof(int)   * BATCH));
    HIP_CALL(hipMalloc(&d_rowEnds,   sizeof(int)   * BATCH));
    HIP_CALL(hipMalloc(&d_seqLens,   sizeof(int)   * BATCH));

    HIP_CALL(hipMemcpy(d_in,        h_in.data(),        sizeof(float) * BATCH * LEN, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_rowStarts, h_rowStarts.data(),  sizeof(int)   * BATCH,       hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_rowEnds,   h_rowEnds.data(),    sizeof(int)   * BATCH,       hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_seqLens,   h_seqLens.data(),    sizeof(int)   * BATCH,       hipMemcpyHostToDevice));

    // ---- Workspace ----
    size_t ws_size_prefill = 0, ws_size_decode = 0;

    aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
        nullptr, ws_size_prefill, nullptr, BATCH, LEN,
        nullptr, nullptr, K, nullptr, nullptr, IS_LARGEST, 0);

    aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
        nullptr, ws_size_decode, nullptr, BATCH, LEN,
        nullptr, nullptr, K, nullptr, nullptr, IS_LARGEST, 0, NEXT_N);

    size_t ws_size = std::max(ws_size_prefill, ws_size_decode);
    void* d_workspace = nullptr;
    HIP_CALL(hipMalloc(&d_workspace, ws_size));
    printf("Workspace: %zu bytes (%.2f KB)\n", ws_size, ws_size / 1024.0);

    {
        int sm_cnt = get_num_cu_func();
        unsigned gd = aiter::calc_grid_dim_v3<float, int, 11, 1024, false, aiter::Phase::Prefill>(
            BATCH, LEN, sm_cnt);
        gd = std::min(gd, 8u);
        gd = std::max(gd, 2u);
        printf("CU count: %d,  grid_dim: %u\n\n", sm_cnt, gd);
    }

    // ---- Timing setup ----
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    auto run_benchmark = [&](const char* label, auto kernel_fn) {
        size_t dummy = 0;
        for(int i = 0; i < WARMUP; ++i) { kernel_fn(dummy); }
        HIP_CALL(hipStreamSynchronize(stream));

        HIP_CALL(hipEventRecord(start, stream));
        for(int i = 0; i < REPEAT; ++i) { kernel_fn(dummy); }
        HIP_CALL(hipEventRecord(stop, stream));
        HIP_CALL(hipEventSynchronize(stop));

        float total_ms = 0;
        HIP_CALL(hipEventElapsedTime(&total_ms, start, stop));
        float avg_us = total_ms / REPEAT * 1000.0f;
        printf("[%s] avg latency: %.2f us  (%.4f ms)  over %d runs\n",
               label, avg_us, total_ms / REPEAT, REPEAT);
    };

    // ---- Benchmark: Prefill ----
    run_benchmark("Prefill", [&](size_t& dummy) {
        aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
            d_workspace, dummy, d_in, BATCH, LEN,
            d_rowStarts, d_rowEnds, K, nullptr, d_out_idx,
            IS_LARGEST, stream);
    });

    // ---- Benchmark: Decode ----
    run_benchmark("Decode", [&](size_t& dummy) {
        aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
            d_workspace, dummy, d_in, BATCH, LEN,
            nullptr, d_seqLens, K, nullptr, d_out_idx,
            IS_LARGEST, stream, NEXT_N);
    });

    // ---- Correctness: Prefill ----
    {
        size_t dummy = 0;
        HIP_CALL(hipMemset(d_out_idx, 0, sizeof(int) * BATCH * K));
        aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Prefill>(
            d_workspace, dummy, d_in, BATCH, LEN,
            d_rowStarts, d_rowEnds, K, nullptr, d_out_idx,
            IS_LARGEST, stream);
        HIP_CALL(hipStreamSynchronize(stream));

        std::vector<int> h_out_idx(BATCH * K);
        HIP_CALL(hipMemcpy(h_out_idx.data(), d_out_idx, sizeof(int) * BATCH * K, hipMemcpyDeviceToHost));

        printf("\n--- Correctness (prefill, batch 0) ---\n");

        std::vector<int> sorted_idx(LEN);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + K, sorted_idx.end(),
                          [&](int a, int b) { return h_in[a] > h_in[b]; });

        float kth_value = h_in[sorted_idx[K - 1]];

        int errors = 0, out_of_range = 0;
        for(int i = 0; i < K; ++i)
        {
            int idx = h_out_idx[i];
            if(idx < 0 || idx >= LEN) { out_of_range++; continue; }
            if(h_in[idx] < kth_value) { errors++; }
        }

        std::vector<float> gpu_vals(K), cpu_vals(K);
        for(int i = 0; i < K; ++i)
        {
            int gpu_idx = h_out_idx[i];
            gpu_vals[i] = (gpu_idx >= 0 && gpu_idx < LEN) ? h_in[gpu_idx] : -FLT_MAX;
            cpu_vals[i] = h_in[sorted_idx[i]];
        }
        std::sort(gpu_vals.begin(), gpu_vals.end(), std::greater<float>());
        std::sort(cpu_vals.begin(), cpu_vals.end(), std::greater<float>());

        bool match = (gpu_vals == cpu_vals);

        printf("  K-th largest (CPU): %f\n", kth_value);
        printf("  Out-of-range:       %d\n", out_of_range);
        printf("  Below k-th:         %d\n", errors);
        printf("  Value set match:    %s\n", match ? "PASS" : "FAIL");
    }

    // ---- Correctness: Decode ----
    {
        size_t dummy = 0;
        HIP_CALL(hipMemset(d_out_idx, 0, sizeof(int) * BATCH * K));
        aiter::standalone_stable_radix_11bits<float, int, false, true, aiter::Phase::Decode>(
            d_workspace, dummy, d_in, BATCH, LEN,
            nullptr, d_seqLens, K, nullptr, d_out_idx,
            IS_LARGEST, stream, NEXT_N);
        HIP_CALL(hipStreamSynchronize(stream));

        std::vector<int> h_out_idx(BATCH * K);
        HIP_CALL(hipMemcpy(h_out_idx.data(), d_out_idx, sizeof(int) * BATCH * K, hipMemcpyDeviceToHost));

        printf("\n--- Correctness (decode, batch 0) ---\n");

        // decode: rowStart=0, rowEnd = seqLen - next_n + 1 = LEN
        int decode_len = h_seqLens[0] - NEXT_N + 1;
        std::vector<int> sorted_idx(decode_len);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::partial_sort(sorted_idx.begin(), sorted_idx.begin() + K, sorted_idx.end(),
                          [&](int a, int b) { return h_in[a] > h_in[b]; });

        float kth_value = h_in[sorted_idx[K - 1]];

        int errors = 0, out_of_range = 0;
        for(int i = 0; i < K; ++i)
        {
            int idx = h_out_idx[i];
            if(idx < 0 || idx >= decode_len) { out_of_range++; continue; }
            if(h_in[idx] < kth_value) { errors++; }
        }

        std::vector<float> gpu_vals(K), cpu_vals(K);
        for(int i = 0; i < K; ++i)
        {
            int gpu_idx = h_out_idx[i];
            gpu_vals[i] = (gpu_idx >= 0 && gpu_idx < decode_len) ? h_in[gpu_idx] : -FLT_MAX;
            cpu_vals[i] = h_in[sorted_idx[i]];
        }
        std::sort(gpu_vals.begin(), gpu_vals.end(), std::greater<float>());
        std::sort(cpu_vals.begin(), cpu_vals.end(), std::greater<float>());

        bool match = (gpu_vals == cpu_vals);

        printf("  Decode len:         %d\n", decode_len);
        printf("  K-th largest (CPU): %f\n", kth_value);
        printf("  Out-of-range:       %d\n", out_of_range);
        printf("  Below k-th:         %d\n", errors);
        printf("  Value set match:    %s\n", match ? "PASS" : "FAIL");
    }

    // ---- Cleanup ----
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
    HIP_CALL(hipStreamDestroy(stream));
    HIP_CALL(hipFree(d_in));
    HIP_CALL(hipFree(d_out_idx));
    HIP_CALL(hipFree(d_rowStarts));
    HIP_CALL(hipFree(d_rowEnds));
    HIP_CALL(hipFree(d_seqLens));
    HIP_CALL(hipFree(d_workspace));

    printf("\nDone.\n");
    return 0;
}
