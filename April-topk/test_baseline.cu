// test_baseline.cu -- Combined standalone benchmark for AIR baseline Top-K
//
// Benchmarks the one-block and multi-block baseline paths extracted from
// `baselline_topk_per_row_kernels.cu` and prints prefill/decode latency for both.
//
// Compile:
//   hipcc -O3 -std=c++17 test_baseline.cu -o test_baseline --offload-arch=gfx942
//
// Run:
//   ./test_baseline [LEN] [K] [BATCH]
//   defaults: LEN=60000  K=2048  BATCH=1

#define aiter aiter_oneblock
#define main baseline_oneblock_internal_main
#define get_num_cu_func get_num_cu_func_oneblock
#include "../topk/topk_baseline_oneblock.cu"
#undef main
#undef get_num_cu_func
#undef aiter
#undef HIP_CALL

#define aiter aiter_multiblock
#define main baseline_multiblock_internal_main
#define get_num_cu_func get_num_cu_func_multiblock
#include "../topk/topk_baseline_multiblocks.cu"
#undef main
#undef get_num_cu_func
#undef aiter

template <typename Fn>
static float run_benchmark(const char* label,
                           int warmup,
                           int repeat,
                           hipStream_t stream,
                           hipEvent_t start,
                           hipEvent_t stop,
                           Fn&& fn)
{
    size_t dummy = 0;
    for(int i = 0; i < warmup; ++i)
    {
        fn(dummy);
    }
    HIP_CALL(hipStreamSynchronize(stream));

    HIP_CALL(hipEventRecord(start, stream));
    for(int i = 0; i < repeat; ++i)
    {
        fn(dummy);
    }
    HIP_CALL(hipEventRecord(stop, stream));
    HIP_CALL(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CALL(hipEventElapsedTime(&total_ms, start, stop));
    float avg_us = total_ms / repeat * 1000.0f;
    printf("[%s] avg latency: %.2f us  (%.4f ms)  over %d runs\n",
           label,
           avg_us,
           total_ms / repeat,
           repeat);
    return avg_us;
}

int main(int argc, char** argv)
{
    const int LEN   = (argc > 1) ? std::atoi(argv[1]) : 60000;
    const int K     = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int BATCH = (argc > 3) ? std::atoi(argv[3]) : 1;

    constexpr int  WARMUP     = 50;
    constexpr int  REPEAT     = 200;
    constexpr bool IS_LARGEST = true;
    constexpr int  NEXT_N     = 1;

    printf("=== Baseline TopK Benchmark ===\n");
    printf("LEN=%d  K=%d  BATCH=%d  WARMUP=%d  REPEAT=%d\n\n", LEN, K, BATCH, WARMUP, REPEAT);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1e6f, 1e6f);

    std::vector<float> h_in(BATCH * LEN);
    for(auto& v : h_in)
    {
        v = dist(rng);
    }

    std::vector<int> h_rowStarts(BATCH, 0);
    std::vector<int> h_rowEnds(BATCH, LEN);
    std::vector<int> h_seqLens(BATCH, LEN);

    float* d_in        = nullptr;
    int*   d_out_idx   = nullptr;
    int*   d_rowStarts = nullptr;
    int*   d_rowEnds   = nullptr;
    int*   d_seqLens   = nullptr;

    HIP_CALL(hipMalloc(&d_in, sizeof(float) * BATCH * LEN));
    HIP_CALL(hipMalloc(&d_out_idx, sizeof(int) * BATCH * K));
    HIP_CALL(hipMalloc(&d_rowStarts, sizeof(int) * BATCH));
    HIP_CALL(hipMalloc(&d_rowEnds, sizeof(int) * BATCH));
    HIP_CALL(hipMalloc(&d_seqLens, sizeof(int) * BATCH));

    HIP_CALL(hipMemcpy(
        d_in, h_in.data(), sizeof(float) * BATCH * LEN, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(
        d_rowStarts, h_rowStarts.data(), sizeof(int) * BATCH, hipMemcpyHostToDevice));
    HIP_CALL(
        hipMemcpy(d_rowEnds, h_rowEnds.data(), sizeof(int) * BATCH, hipMemcpyHostToDevice));
    HIP_CALL(
        hipMemcpy(d_seqLens, h_seqLens.data(), sizeof(int) * BATCH, hipMemcpyHostToDevice));

    size_t ws_oneblock_prefill  = 0;
    size_t ws_oneblock_decode   = 0;
    size_t ws_multiblock_prefill = 0;
    size_t ws_multiblock_decode  = 0;

    aiter_oneblock::standalone_stable_radix_11bits<float, int, false, true, aiter_oneblock::Phase::Prefill>(
        nullptr,
        ws_oneblock_prefill,
        nullptr,
        BATCH,
        LEN,
        nullptr,
        nullptr,
        K,
        nullptr,
        nullptr,
        IS_LARGEST,
        0);
    aiter_oneblock::standalone_stable_radix_11bits<float, int, false, true, aiter_oneblock::Phase::Decode>(
        nullptr,
        ws_oneblock_decode,
        nullptr,
        BATCH,
        LEN,
        nullptr,
        nullptr,
        K,
        nullptr,
        nullptr,
        IS_LARGEST,
        0,
        NEXT_N);

    aiter_multiblock::standalone_stable_radix_11bits<float, int, false, true, aiter_multiblock::Phase::Prefill>(
        nullptr,
        ws_multiblock_prefill,
        nullptr,
        BATCH,
        LEN,
        nullptr,
        nullptr,
        K,
        nullptr,
        nullptr,
        IS_LARGEST,
        0);
    aiter_multiblock::standalone_stable_radix_11bits<float, int, false, true, aiter_multiblock::Phase::Decode>(
        nullptr,
        ws_multiblock_decode,
        nullptr,
        BATCH,
        LEN,
        nullptr,
        nullptr,
        K,
        nullptr,
        nullptr,
        IS_LARGEST,
        0,
        NEXT_N);

    size_t ws_oneblock  = std::max(ws_oneblock_prefill, ws_oneblock_decode);
    size_t ws_multiblock = std::max(ws_multiblock_prefill, ws_multiblock_decode);

    void* d_workspace_oneblock  = nullptr;
    void* d_workspace_multiblock = nullptr;
    HIP_CALL(hipMalloc(&d_workspace_oneblock, ws_oneblock));
    HIP_CALL(hipMalloc(&d_workspace_multiblock, ws_multiblock));

    int sm_cnt = get_num_cu_func_multiblock();
    unsigned gd_prefill =
        aiter_multiblock::calc_grid_dim<float, int, 11, 1024, false, aiter_multiblock::Phase::Prefill>(
            BATCH, LEN, sm_cnt);
    unsigned gd_decode =
        aiter_multiblock::calc_grid_dim<float, int, 11, 1024, false, aiter_multiblock::Phase::Decode>(
            BATCH, LEN, sm_cnt);

    printf("One-block workspace:  %zu bytes (%.2f KB)\n", ws_oneblock, ws_oneblock / 1024.0);
    printf("Multi-block workspace:%zu bytes (%.2f KB)\n", ws_multiblock, ws_multiblock / 1024.0);
    printf("CU count: %d, multi-block gd(prefill)=%u, gd(decode)=%u\n\n",
           sm_cnt,
           gd_prefill,
           gd_decode);

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    hipStream_t stream;
    HIP_CALL(hipStreamCreate(&stream));

    printf("---- One-block ----\n");
    float oneblock_prefill_us = run_benchmark(
        "oneblock prefill", WARMUP, REPEAT, stream, start, stop, [&](size_t& dummy) {
            aiter_oneblock::standalone_stable_radix_11bits<float, int, false, true, aiter_oneblock::Phase::Prefill>(
                d_workspace_oneblock,
                dummy,
                d_in,
                BATCH,
                LEN,
                d_rowStarts,
                d_rowEnds,
                K,
                nullptr,
                d_out_idx,
                IS_LARGEST,
                stream);
        });
    float oneblock_decode_us = run_benchmark(
        "oneblock decode", WARMUP, REPEAT, stream, start, stop, [&](size_t& dummy) {
            aiter_oneblock::standalone_stable_radix_11bits<float, int, false, true, aiter_oneblock::Phase::Decode>(
                d_workspace_oneblock,
                dummy,
                d_in,
                BATCH,
                LEN,
                nullptr,
                d_seqLens,
                K,
                nullptr,
                d_out_idx,
                IS_LARGEST,
                stream,
                NEXT_N);
        });

    printf("\n---- Multi-block ----\n");
    float multiblock_prefill_us = run_benchmark(
        "multiblock prefill", WARMUP, REPEAT, stream, start, stop, [&](size_t& dummy) {
            aiter_multiblock::standalone_stable_radix_11bits<float, int, false, true, aiter_multiblock::Phase::Prefill>(
                d_workspace_multiblock,
                dummy,
                d_in,
                BATCH,
                LEN,
                d_rowStarts,
                d_rowEnds,
                K,
                nullptr,
                d_out_idx,
                IS_LARGEST,
                stream);
        });
    float multiblock_decode_us = run_benchmark(
        "multiblock decode", WARMUP, REPEAT, stream, start, stop, [&](size_t& dummy) {
            aiter_multiblock::standalone_stable_radix_11bits<float, int, false, true, aiter_multiblock::Phase::Decode>(
                d_workspace_multiblock,
                dummy,
                d_in,
                BATCH,
                LEN,
                nullptr,
                d_seqLens,
                K,
                nullptr,
                d_out_idx,
                IS_LARGEST,
                stream,
                NEXT_N);
        });

    printf("\nSummary (avg us)\n");
    printf("  oneblock  prefill:   %.2f\n", oneblock_prefill_us);
    printf("  oneblock  decode:    %.2f\n", oneblock_decode_us);
    printf("  multiblock prefill:  %.2f\n", multiblock_prefill_us);
    printf("  multiblock decode:   %.2f\n", multiblock_decode_us);

    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
    HIP_CALL(hipStreamDestroy(stream));
    HIP_CALL(hipFree(d_in));
    HIP_CALL(hipFree(d_out_idx));
    HIP_CALL(hipFree(d_rowStarts));
    HIP_CALL(hipFree(d_rowEnds));
    HIP_CALL(hipFree(d_seqLens));
    HIP_CALL(hipFree(d_workspace_oneblock));
    HIP_CALL(hipFree(d_workspace_multiblock));

    printf("\nDone.\n");
    return 0;
}
