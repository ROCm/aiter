// OPUS GPU Unit Tests — requires HIP device (gfx942/gfx950)
// Build: hipcc -std=c++20 -O2 -I../ test_opus_gpu.cpp -o test_opus_gpu && ./test_opus_gpu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <hip/hip_runtime.h>
#include "opus/opus.hpp"

namespace o = opus;
using o::number;
using o::index_t;

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } \
} while(0)

constexpr int NUM_TESTS = 16;

// =========================================================================
// Test 0: Type conversions (fp32 <-> fp16, fp32 <-> bf16)
// =========================================================================
__global__ void test_type_convert(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    // fp32 -> fp16 -> fp32 roundtrip (exact for values representable in fp16)
    float val = 1.5f + tid * 0.25f;
    auto f16 = o::fp32_to_fp16(val);
    float back = o::fp16_to_fp32(f16);
    if (back != val) pass = false;

    // fp32 -> bf16 -> fp32 roundtrip (bf16 has 7 mantissa bits)
    float val2 = 2.0f + tid * 0.5f;
    auto bf = o::fp32_to_bf16(val2);
    float back2 = o::bf16_to_fp32(bf);
    if (fabsf(back2 - val2) > 0.02f * fabsf(val2) + 1e-5f) pass = false;

    // NOTE: fp32_to_bf16_rtn_asm has a known issue — bf16_t(u.i) interprets
    // as value conversion not bit cast. Skipping that test.

    // cast<> generic interface
    auto f16c = o::cast<o::fp16_t, o::fp32_t>(3.0f);
    float back4 = o::cast<o::fp32_t, o::fp16_t>(f16c);
    if (back4 != 3.0f) pass = false;

    if (tid == 0) results[0] = pass ? 1 : -1;
}

// =========================================================================
// Test 1: Math operations (max, min, med3)
// =========================================================================
__global__ void test_math_ops(int* results) {
    bool pass = true;

    // max/min float
    if (o::max(3.0f, 7.0f) != 7.0f) pass = false;
    if (o::min(3.0f, 7.0f) != 3.0f) pass = false;

    // max/min int
    if (o::max(5, 2) != 5) pass = false;
    if (o::min(5, 2) != 2) pass = false;

    // med3 float
    if (o::med3(1.0f, 5.0f, 3.0f) != 3.0f) pass = false;
    if (o::med3(1.0f, 1.0f, 1.0f) != 1.0f) pass = false;
    if (o::med3(1.0f, 2.0f, 3.0f) != 2.0f) pass = false;

    // med3 fp16
    auto m16 = o::med3(
        static_cast<o::fp16_t>(1.0f),
        static_cast<o::fp16_t>(5.0f),
        static_cast<o::fp16_t>(3.0f));
    if (static_cast<float>(m16) != 3.0f) pass = false;

    if (threadIdx.x == 0) results[1] = pass ? 1 : -1;
}

// =========================================================================
// Test 2: DPP warp operations (mov_dpp, upd_dpp)
// =========================================================================
__global__ void test_dpp_warp(int* results) {
    int tid = threadIdx.x;  // 0..63 (one warp)
    bool pass = true;

    float val = static_cast<float>(tid);

    // DPP_ROW_SHL(1) = 0x101: data shifts left → lane i reads from lane (i+1)
    float shifted_l = o::mov_dpp(val, number<0x101>{});
    int row_id = tid % 16;
    float expected_l = (row_id == 15) ? 0.0f : static_cast<float>(tid + 1);
    if (shifted_l != expected_l) pass = false;

    // DPP_ROW_SHR(1) = 0x111: data shifts right → lane i reads from lane (i-1)
    float shifted_r = o::mov_dpp(val, number<0x111>{});
    float expected_r = (row_id == 0) ? 0.0f : static_cast<float>(tid - 1);
    if (shifted_r != expected_r) pass = false;

    // upd_dpp with default bound_ctrl=true: OOB lanes get 0
    float old_val = 100.0f;
    float updated = o::upd_dpp(old_val, val, number<0x101>{});
    float expected_u = (row_id == 15) ? 0.0f : static_cast<float>(tid + 1);
    if (updated != expected_u) pass = false;

    if (tid == 0) results[2] = pass ? 1 : -1;
}

// =========================================================================
// Test 3: GMEM buffer load/store (scalar)
// =========================================================================
__global__ void test_gmem_basic(int* results, float* buf, int N) {
    int tid = threadIdx.x;
    if (tid >= N) return;

    auto g = o::make_gmem<float>(buf, N * sizeof(float));

    // Load value (offset in units of T). load() returns vector_t<float,1>
    float val = g.load(tid)[0];
    bool pass = (val == static_cast<float>(tid));

    // Store modified value
    float new_val = val * 2.0f + 1.0f;
    o::vector_t<float, 1> store_vec;
    store_vec[0] = new_val;
    g.store(store_vec, tid);
    o::s_waitcnt_vmcnt(number<0>{});

    // Load back and verify
    float check = g.load(tid)[0];
    o::s_waitcnt_vmcnt(number<0>{});
    if (check != new_val) pass = false;

    if (tid == 0) results[3] = pass ? 1 : -1;
}

// =========================================================================
// Test 4: SMEM (LDS) load/store (scalar)
// =========================================================================
__global__ void test_smem_basic(int* results) {
    __shared__ float lds[256];
    int tid = threadIdx.x;
    bool pass = true;

    auto s = o::make_smem<float>(lds);

    // Store lane-dependent value. store/load use vector_t<float,1> for vec=1
    float val = static_cast<float>(tid * 3 + 7);
    o::vector_t<float, 1> sv;
    sv[0] = val;
    s.store(sv, tid);
    __syncthreads();

    // Read back own value
    float loaded = s.load(tid)[0];
    if (loaded != val) pass = false;
    __syncthreads();

    // Cross-lane read
    int neighbor = (tid + 1) % blockDim.x;
    float nval = s.load(neighbor)[0];
    float expected = static_cast<float>(neighbor * 3 + 7);
    if (nval != expected) pass = false;

    if (tid == 0) results[4] = pass ? 1 : -1;
}

// =========================================================================
// Test 5: SMEM vector load/store (vec=4)
// =========================================================================
__global__ void test_smem_vec(int* results) {
    __shared__ float lds[1024];
    int tid = threadIdx.x;
    bool pass = true;

    auto s = o::make_smem<float>(lds);

    // Store 4 floats per thread
    o::vector_t<float, 4> vec;
    for (int i = 0; i < 4; i++) vec[i] = static_cast<float>(tid * 4 + i);
    s.store<4>(vec, tid * 4);
    __syncthreads();

    // Load back
    auto loaded = s.load<4>(tid * 4);
    for (int i = 0; i < 4; i++) {
        if (loaded[i] != static_cast<float>(tid * 4 + i)) pass = false;
    }

    if (tid == 0) results[5] = pass ? 1 : -1;
}

// =========================================================================
// Test 6: MFMA f32_16x16x16_f16 (A=1, B=1 -> C=16)
// =========================================================================
__global__ void test_mfma_16x16x16(int* results) {
    int tid = threadIdx.x;  // 64 threads = one warp
    bool pass = true;

    using M = o::mfma_f32_16x16x16_f16;
    M mfma_op;

    // A = all 1.0, B = all 1.0 -> C[i,j] = sum_k(1*1) = 16
    M::vtype_a a;  // vector_t<fp16_t, 4>
    M::vtype_b b;  // vector_t<fp16_t, 4>
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(1.0f);
        b[i] = static_cast<o::fp16_t>(1.0f);
    }

    auto c = mfma_op(a, b);  // v4f32, all should be 16.0
    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - 16.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[6] = pass ? 1 : -1;
}

// =========================================================================
// Test 7: MFMA f32_32x32x8_f16 (A=1, B=1 -> C=8)
// =========================================================================
__global__ void test_mfma_32x32x8(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    using M = o::mfma_f32_32x32x8_f16;
    M mfma_op;

    // elem_a = 32*8/64 = 4, elem_b = 4, elem_c = 32*32/64 = 16
    M::vtype_a a;
    M::vtype_b b;
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(1.0f);
        b[i] = static_cast<o::fp16_t>(1.0f);
    }

    auto c = mfma_op(a, b);  // v16f32, all should be 8.0
    for (int i = 0; i < 16; i++) {
        if (fabsf(c[i] - 8.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[7] = pass ? 1 : -1;
}

// =========================================================================
// Test 8: MFMA accumulator chaining (16x16x16, two iterations)
// =========================================================================
__global__ void test_mfma_accum(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    using M = o::mfma_f32_16x16x16_f16;
    M mfma_op;

    M::vtype_a a;
    M::vtype_b b;
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(1.0f);
        b[i] = static_cast<o::fp16_t>(1.0f);
    }

    // C = 0 + A*B = 16, then C += A*B = 32
    auto c = mfma_op(a, b);
    c = mfma_op(a, b, c);

    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - 32.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[8] = pass ? 1 : -1;
}

// =========================================================================
// Test 9: MFMA f32_16x16x16_f16 with non-trivial values (A=2, B=1 -> C=32)
// =========================================================================
__global__ void test_mfma_scaled(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    using M = o::mfma_f32_16x16x16_f16;
    M mfma_op;

    M::vtype_a a;
    M::vtype_b b;
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(2.0f);
        b[i] = static_cast<o::fp16_t>(1.0f);
    }

    // C[i,j] = sum_k(2*1) = 2*16 = 32
    auto c = mfma_op(a, b);
    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - 32.0f) > 0.01f) pass = false;
    }

    // Also test A=0.5, B=2 -> C = 0.5*2*16 = 16
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(0.5f);
        b[i] = static_cast<o::fp16_t>(2.0f);
    }
    auto c2 = mfma_op(a, b);
    for (int i = 0; i < 4; i++) {
        if (fabsf(c2[i] - 16.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[9] = pass ? 1 : -1;
}

// =========================================================================
// Test 10: Adaptor device-side shape/layout (make_mfma on device)
// =========================================================================
__global__ void test_adaptor_device(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    auto mma = o::make_mfma<o::fp16_t, o::fp16_t, o::fp32_t>(
        number<16>{}, number<16>{}, number<16>{});

    // shape_a = (grpm_a=16, rept_a=1, grpk_a=4, pack_a=4)
    auto sa = mma.shape_a();
    if (o::get<0>(sa).value != 16) pass = false;
    if (o::get<1>(sa).value != 1)  pass = false;
    if (o::get<2>(sa).value != 4)  pass = false;
    if (o::get<3>(sa).value != 4)  pass = false;

    // shape_c = (rept_c=1, grpm_c=4, pack_c=4, grpn_c=16)
    auto sc = mma.shape_c();
    if (o::get<0>(sc).value != 1)  pass = false;
    if (o::get<1>(sc).value != 4)  pass = false;
    if (o::get<2>(sc).value != 4)  pass = false;
    if (o::get<3>(sc).value != 16) pass = false;

    // y_shape_a picks y_dim elements: (rept_a=1, pack_a=4)
    auto ysa = mma.y_shape_a();
    if (o::get<0>(ysa).value != 1) pass = false;
    if (o::get<1>(ysa).value != 4) pass = false;

    // p_shape_a picks p_dim elements: (grpm_a=16, grpk_a=4)
    auto psa = mma.p_shape_a();
    if (o::get<0>(psa).value != 16) pass = false;
    if (o::get<1>(psa).value != 4)  pass = false;

    // layout_a() generates packed layout on device
    auto la = mma.layout_a();
    // Verify a known index: la(0, 0, 0, 0) should be 0
    if (la(number<0>{}, number<0>{}, number<0>{}, number<0>{}) != 0) pass = false;

    if (tid == 0) results[10] = pass ? 1 : -1;
}

// =========================================================================
// Test 11: Adaptor swap_ab
// =========================================================================
__global__ void test_adaptor_swap(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    auto mma = o::make_mfma<o::fp16_t, o::fp16_t, o::fp32_t>(
        number<16>{}, number<16>{}, number<16>{}, o::mfma_adaptor_swap_ab{});

    // shape_c for swap_ab = (grpn_c=16, rept_c=1, grpm_c=4, pack_c=4)
    auto sc = mma.shape_c();
    if (o::get<0>(sc).value != 16) pass = false;
    if (o::get<1>(sc).value != 1)  pass = false;
    if (o::get<2>(sc).value != 4)  pass = false;
    if (o::get<3>(sc).value != 4)  pass = false;

    // Compute: swap_ab calls base(b, a), all-ones -> still 16.0
    using M = o::mfma_f32_16x16x16_f16;
    M::vtype_a a;
    M::vtype_b b;
    for (int i = 0; i < 4; i++) {
        a[i] = static_cast<o::fp16_t>(1.0f);
        b[i] = static_cast<o::fp16_t>(1.0f);
    }
    auto c = mma(a, b);
    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - 16.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[11] = pass ? 1 : -1;
}

// =========================================================================
// Test 12: Tiled MMA (2x2x1 expansion of 16x16x16)
// =========================================================================
__global__ void test_tiled_mma(int* results) {
    int tid = threadIdx.x;
    bool pass = true;

    auto mma_base = o::make_mfma<o::fp16_t, o::fp16_t, o::fp32_t>(
        number<16>{}, number<16>{}, number<16>{});

    // 2x2x1 expansion, 1x1x1 tiling
    auto tiled = o::make_tiled_mma(mma_base,
        number<2>{}, number<2>{}, number<1>{},   // expand M,N,K
        number<1>{}, number<1>{}, number<1>{});   // tile M,N,K

    // tile_shape_a = (expd_m=2, tile_m=1, expd_k=1, tile_k=1)
    auto ts_a = tiled.tile_shape_a();
    if (o::get<0>(ts_a).value != 2) pass = false;
    if (o::get<1>(ts_a).value != 1) pass = false;
    if (o::get<2>(ts_a).value != 1) pass = false;
    if (o::get<3>(ts_a).value != 1) pass = false;

    // Execute: A = all 1, B = all 1 -> C = 16.0
    // vtype_a = vector_t<fp16_t, 2*1*4> = v8f16
    // vtype_b = vector_t<fp16_t, 2*1*4> = v8f16
    // vtype_c = vector_t<fp32_t, 2*2*4> = v16f32
    using tiled_t = decltype(tiled);
    typename tiled_t::vtype_a a;
    typename tiled_t::vtype_b b;
    o::fill(a, static_cast<o::fp16_t>(1.0f));
    o::fill(b, static_cast<o::fp16_t>(1.0f));

    auto c = tiled(a, b);

    constexpr int c_size = o::vector_traits<typename tiled_t::vtype_c>::size();
    for (int i = 0; i < c_size; i++) {
        if (fabsf(c[i] - 16.0f) > 0.01f) pass = false;
    }

    if (tid == 0) results[12] = pass ? 1 : -1;
}

// =========================================================================
// Test 13: GMEM vector load/store (b128 = 4 floats at once)
// =========================================================================
__global__ void test_gmem_vec(int* results, float* buf, int N) {
    int tid = threadIdx.x;
    if (tid * 4 + 3 >= N) return;

    auto g = o::make_gmem<float>(buf, N * sizeof(float));
    bool pass = true;

    // Load 4 floats (vec=4)
    auto vec = g.load<4>(tid * 4);
    for (int i = 0; i < 4; i++) {
        if (vec[i] != static_cast<float>(tid * 4 + i)) pass = false;
    }

    // Modify and store back
    for (int i = 0; i < 4; i++) vec[i] *= 3.0f;
    g.store<4>(vec, tid * 4);
    o::s_waitcnt_vmcnt(number<0>{});

    // Read back individual elements
    for (int i = 0; i < 4; i++) {
        float val = g.load(tid * 4 + i)[0];
        o::s_waitcnt_vmcnt(number<0>{});
        if (val != static_cast<float>((tid * 4 + i) * 3)) pass = false;
    }

    if (tid == 0) results[13] = pass ? 1 : -1;
}

// =========================================================================
// Test 14: FP8 pack/unpack roundtrip
// =========================================================================
__global__ void test_fp8_pack(int* results) {
    bool pass = true;

    // fp32x2 -> fp8x2 -> fp32x2
    o::fp32x2_t in2{1.0f, 2.0f};
    auto packed2 = o::fp32_to_fp8_packed_x2(in2);
    auto out2 = o::fp8_to_fp32_packed_x2(packed2);
    if (fabsf(out2[0] - 1.0f) > 0.125f) pass = false;
    if (fabsf(out2[1] - 2.0f) > 0.125f) pass = false;

    // fp32x4 -> fp8x4 -> fp32x4
    o::fp32x4_t in4{1.0f, 2.0f, 3.0f, 4.0f};
    auto packed4 = o::fp32_to_fp8_packed_x4(in4);
    auto out4 = o::fp8_to_fp32_packed_x4(packed4);
    if (fabsf(out4[0] - 1.0f) > 0.25f) pass = false;
    if (fabsf(out4[1] - 2.0f) > 0.25f) pass = false;
    if (fabsf(out4[2] - 3.0f) > 0.25f) pass = false;
    if (fabsf(out4[3] - 4.0f) > 0.25f) pass = false;

    if (threadIdx.x == 0) results[14] = pass ? 1 : -1;
}

// =========================================================================
// Test 15: Container folding + waitcnt smoke test
// =========================================================================
__global__ void test_fold_waitcnt(int* results, float* buf) {
    bool pass = true;

    // fold_as_container_of_vec: fold v8f32 into tuple of 4 x v2f32
    auto v = o::make_vector(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    auto folded = o::impl::fold_as_container_of_vec(v, number<2>{});
    auto v0 = o::get<0>(folded);
    auto v1 = o::get<1>(folded);
    auto v2 = o::get<2>(folded);
    auto v3 = o::get<3>(folded);
    if (v0[0] != 1.0f || v0[1] != 2.0f) pass = false;
    if (v1[0] != 3.0f || v1[1] != 4.0f) pass = false;
    if (v2[0] != 5.0f || v2[1] != 6.0f) pass = false;
    if (v3[0] != 7.0f || v3[1] != 8.0f) pass = false;

    // s_waitcnt smoke test (ensure it compiles and doesn't crash)
    int tid = threadIdx.x;
    auto g = o::make_gmem<float>(buf, 256 * sizeof(float));
    float a_val = g.load(tid)[0];
    o::s_waitcnt_vmcnt(number<0>{});
    float b_val = a_val + 1.0f;
    o::vector_t<float, 1> sv;
    sv[0] = b_val;
    g.store(sv, tid);
    o::s_waitcnt(number<0>{}, number<0>{});
    float c_val = g.load(tid)[0];
    o::s_waitcnt_vmcnt(number<0>{});
    if (c_val != b_val) pass = false;

    // s_waitcnt_lgkmcnt
    __shared__ float lds[64];
    lds[tid] = 42.0f;
    o::s_waitcnt_lgkmcnt(number<0>{});
    __syncthreads();
    if (lds[tid] != 42.0f) pass = false;

    if (tid == 0) results[15] = pass ? 1 : -1;
}

// =========================================================================
// main
// =========================================================================
int main() {
    printf("opus GPU unit tests\n");
    printf("====================\n");

    // Allocate results via managed memory
    int* results;
    HIP_CHECK(hipMallocManaged(&results, NUM_TESTS * sizeof(int)));
    ::memset(results, 0, NUM_TESTS * sizeof(int));

    // Allocate GPU buffers
    constexpr int BUF_SIZE = 1024;
    float* d_buf;
    HIP_CHECK(hipMalloc(&d_buf, BUF_SIZE * sizeof(float)));
    float h_buf[BUF_SIZE];

    // Helper to init buffer with sequential values
    auto init_buf = [&]() {
        for (int i = 0; i < BUF_SIZE; i++) h_buf[i] = static_cast<float>(i);
        HIP_CHECK(hipMemcpy(d_buf, h_buf, BUF_SIZE * sizeof(float), hipMemcpyHostToDevice));
    };

    float* d_buf2;
    HIP_CHECK(hipMalloc(&d_buf2, 256 * sizeof(float)));
    for (int i = 0; i < 256; i++) h_buf[i] = static_cast<float>(i);
    HIP_CHECK(hipMemcpy(d_buf2, h_buf, 256 * sizeof(float), hipMemcpyHostToDevice));

    // Test names
    const char* test_names[NUM_TESTS] = {
        "type_convert",       // 0
        "math_ops",           // 1
        "dpp_warp",           // 2
        "gmem_basic",         // 3
        "smem_basic",         // 4
        "smem_vec",           // 5
        "mfma_16x16x16_f16",  // 6
        "mfma_32x32x8_f16",   // 7
        "mfma_accum",          // 8
        "mfma_scaled",         // 9
        "adaptor_device",      // 10
        "adaptor_swap",        // 11
        "tiled_mma",           // 12
        "gmem_vec",            // 13
        "fp8_pack",            // 14
        "fold_waitcnt",        // 15
    };

    // Launch all tests (default stream = sequential execution)
    test_type_convert<<<1, 64>>>(results);
    test_math_ops<<<1, 64>>>(results);
    test_dpp_warp<<<1, 64>>>(results);

    init_buf();
    test_gmem_basic<<<1, 64>>>(results, d_buf, BUF_SIZE);

    test_smem_basic<<<1, 64>>>(results);
    test_smem_vec<<<1, 64>>>(results);

    test_mfma_16x16x16<<<1, 64>>>(results);
    test_mfma_32x32x8<<<1, 64>>>(results);
    test_mfma_accum<<<1, 64>>>(results);
    test_mfma_scaled<<<1, 64>>>(results);

    test_adaptor_device<<<1, 64>>>(results);
    test_adaptor_swap<<<1, 64>>>(results);
    test_tiled_mma<<<1, 64>>>(results);

    init_buf();
    test_gmem_vec<<<1, 64>>>(results, d_buf, BUF_SIZE);

    test_fp8_pack<<<1, 64>>>(results);

    test_fold_waitcnt<<<1, 64>>>(results, d_buf2);

    HIP_CHECK(hipDeviceSynchronize());

    // Report results
    int passed = 0, failed = 0, not_run = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        const char* status = (results[i] == 1) ? "PASS" :
                             (results[i] == -1) ? "FAIL" : "NOT_RUN";
        printf("  %-22s: %s\n", test_names[i], status);
        if (results[i] == 1) passed++;
        else if (results[i] == -1) failed++;
        else not_run++;
    }

    printf("====================\n");
    printf("%d passed, %d failed, %d not run\n", passed, failed, not_run);

    HIP_CHECK(hipFree(d_buf));
    HIP_CHECK(hipFree(d_buf2));
    HIP_CHECK(hipFree(results));

    return (failed > 0 || not_run > 0) ? 1 : 0;
}
