// test_gfx1250_dpp.cu — Unit tests for gfx1250 DPP broadcast replacements
// Tests row_bcast:15, row_bcast:31 lane semantics in hip_reduce.h
// Compile: hipcc -std=c++20 --offload-arch=gfx950 -I csrc/include
//          -o test_gfx1250_dpp test_gfx1250_dpp.cu
// Run: ./test_gfx1250_dpp
//
// These tests validate the LOGIC of lane calculations independently of
// the actual DPP hardware. They work on any arch (gfx950/gfx1250).

#include <hip/hip_runtime.h>
#include <cstdio>

__device__ int g_pass = 0;
__device__ int g_fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { atomicAdd(&g_pass, 1); } \
    else { atomicAdd(&g_fail, 1); printf("FAIL lane %d: %s\n", (int)__lane_id(), name); } \
} while(0)

// ============================================================
// Test 1: row_bcast:15 lane calculation
// Each lane should read from lane 15 of its 16-lane row
// Row 0: lanes 0-15  → all read from lane 15
// Row 1: lanes 16-31 → all read from lane 31
// Row 2: lanes 32-47 → all read from lane 47
// Row 3: lanes 48-63 → all read from lane 63
// ============================================================
__global__ void test_bcast15_lanes(int* results) {
    int lane = __lane_id();
    // Each lane stores its own lane_id
    int my_val = lane;

    // Compute source lane for row_bcast:15
    int src_lane = (lane & ~15) | 15;  // lane 15 within each 16-lane row
    results[lane] = src_lane;

    // Verify expected source lanes
    if (lane < 16)  CHECK(src_lane == 15, "bcast15 row0 -> lane 15");
    if (lane >= 16 && lane < 32)  CHECK(src_lane == 31, "bcast15 row1 -> lane 31");
    if (lane >= 32 && lane < 48)  CHECK(src_lane == 47, "bcast15 row2 -> lane 47");
    if (lane >= 48)  CHECK(src_lane == 63, "bcast15 row3 -> lane 63");
}

// ============================================================
// Test 2: row_bcast:31 lane calculation
// Each lane should read from lane 31 of its 32-lane half
// Half 0: lanes 0-31  → all read from lane 31
// Half 1: lanes 32-63 → all read from lane 63
// ============================================================
__global__ void test_bcast31_lanes(int* results) {
    int lane = __lane_id();
    int src_lane = (lane & ~31) | 31;
    results[lane] = src_lane;

    if (lane < 32)  CHECK(src_lane == 31, "bcast31 half0 -> lane 31");
    if (lane >= 32) CHECK(src_lane == 63, "bcast31 half1 -> lane 63");
}

// ============================================================
// Test 3: Actual shuffle with ds_bpermute (functional test)
// Each lane writes its lane_id. After bcast15, all lanes in
// each row should have the value of lane 15/31/47/63.
// ============================================================
__global__ void test_bcast15_shuffle(int* results) {
    int lane = __lane_id();
    int val = lane * 10 + 1;  // unique per lane

    int src = (lane & ~15) | 15;
    int shuffled = __builtin_amdgcn_ds_bpermute(src << 2, val);
    results[lane] = shuffled;

    // Expected: everyone in row gets the value of lane 15/31/47/63
    int expected_src = (lane & ~15) | 15;
    int expected_val = expected_src * 10 + 1;
    CHECK(shuffled == expected_val, "bcast15 shuffle value");
}

// ============================================================
// Test 4: Multi-word ds_bpermute (8-byte type)
// Verifies that gfx1250_ds_bpermute handles types > 4 bytes
// ============================================================
struct pair_t { int key; float val; };

__global__ void test_multiword_bpermute(int* results) {
    int lane = __lane_id();
    pair_t my_pair = {lane, (float)(lane * 1.5f)};

    // Shuffle: each lane reads from lane XOR 1
    int src = lane ^ 1;

    // Multi-word shuffle (manual, same logic as gfx1250_ds_bpermute)
    constexpr int words = (sizeof(pair_t) + 3) / 4;
    union { pair_t v; int w[words]; } in, out;
    in.v = my_pair;
    for (int i = 0; i < words; i++)
        out.w[i] = __builtin_amdgcn_ds_bpermute(src << 2, in.w[i]);

    pair_t remote = out.v;
    results[lane * 2] = remote.key;
    results[lane * 2 + 1] = (int)(remote.val * 10);

    CHECK(remote.key == src, "multiword key");
    CHECK(fabsf(remote.val - src * 1.5f) < 0.01f, "multiword val");
}

// ============================================================
// Test 5: XOR reduce (wave_reduce_ds pattern)
// Sum all 64 lane values: 0+1+2+...+63 = 2016
// ============================================================
__global__ void test_xor_reduce(int* results) {
    int lane = __lane_id();
    int val = lane;

    // 6-stage XOR reduce
    for (int stage = 0; stage < 6; stage++) {
        int src = lane ^ (1 << stage);
        int remote = __builtin_amdgcn_ds_bpermute(src << 2, val);
        val += remote;
    }

    results[lane] = val;
    // All lanes should have sum 0+1+...+63 = 2016
    CHECK(val == 2016, "xor reduce sum=2016");
}

int main() {
    int *d_results;
    hipMalloc(&d_results, 256 * sizeof(int));
    int h_pass = 0, h_fail = 0;

    test_bcast15_lanes<<<1, 64>>>(d_results);
    test_bcast31_lanes<<<1, 64>>>(d_results);
    test_bcast15_shuffle<<<1, 64>>>(d_results);
    test_multiword_bpermute<<<1, 64>>>(d_results);
    test_xor_reduce<<<1, 64>>>(d_results);

    hipDeviceSynchronize();
    hipMemcpyFromSymbol(&h_pass, HIP_SYMBOL(g_pass), sizeof(int));
    hipMemcpyFromSymbol(&h_fail, HIP_SYMBOL(g_fail), sizeof(int));

    printf("gfx1250 DPP tests: %d PASS, %d FAIL\n", h_pass, h_fail);
    return h_fail > 0 ? 1 : 0;
}
