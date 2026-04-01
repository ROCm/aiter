// test_ck_tile_shim.cu — Unit tests for ck_tile_shim.h
// Compile: hipcc -std=c++20 --offload-arch=gfx950 -DDISABLE_CK=1 -DAITER_CK_FREE=1
//          -I csrc/include -I csrc/include/ck_tile -I csrc/include/opus
//          -o test_ck_tile_shim test_ck_tile_shim.cu
// Run: ./test_ck_tile_shim
// Expected: all PASS, 0 FAIL

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include "aiter_hip_common.h"
#include "ck_tile/vec_convert.h"

__device__ int g_pass = 0;
__device__ int g_fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { atomicAdd(&g_pass, 1); } \
    else { atomicAdd(&g_fail, 1); printf("FAIL: %s\n", name); } \
} while(0)

#define CHECK_EQ(a, b, name) CHECK((a) == (b), name)
#define CHECK_NEAR(a, b, eps, name) CHECK(fabsf((float)(a) - (float)(b)) < (eps), name)

// ============================================================
// Test 1: constant<v> arithmetic operators
// ============================================================
__global__ void test_constant_ops() {
    using namespace ck_tile;
    // Basic arithmetic
    CHECK_EQ((number<3>{} + number<4>{}).value, 7, "const 3+4=7");
    CHECK_EQ((number<10>{} - number<3>{}).value, 7, "const 10-3=7");
    CHECK_EQ((number<3>{} * number<4>{}).value, 12, "const 3*4=12");
    CHECK_EQ((number<12>{} / number<4>{}).value, 3, "const 12/4=3");
    CHECK_EQ((number<7>{} % number<3>{}).value, 1, "const 7%3=1");
    // Comparison
    CHECK_EQ((number<3>{} == number<3>{}).value, true, "const 3==3");
    CHECK_EQ((number<3>{} != number<4>{}).value, true, "const 3!=4");
    CHECK_EQ((number<3>{} < number<4>{}).value, true, "const 3<4");
    CHECK_EQ((number<4>{} > number<3>{}).value, true, "const 4>3");
    // Bitwise
    CHECK_EQ((number<0xFF>{} & number<0x0F>{}).value, 0x0F, "const &");
    CHECK_EQ((number<0xF0>{} | number<0x0F>{}).value, 0xFF, "const |");
    // Implicit conversion
    int x = number<42>{};
    CHECK_EQ(x, 42, "const implicit conv");
}

// ============================================================
// Test 2: vector_traits
// ============================================================
__global__ void test_vector_traits() {
    using namespace ck_tile;
    // thread_buffer
    CHECK_EQ((vector_traits<thread_buffer<float, 4>>::vector_size), 4, "vt tb<f,4>=4");
    CHECK_EQ((vector_traits<thread_buffer<fp8_t, 8>>::vector_size), 8, "vt tb<fp8,8>=8");
    // ext_vector_type
    using f32x4 = float __attribute__((ext_vector_type(4)));
    using f16x8 = fp16_t __attribute__((ext_vector_type(8)));
    CHECK_EQ((vector_traits<f32x4>::vector_size), 4, "vt ext f32x4=4");
    CHECK_EQ((vector_traits<f16x8>::vector_size), 8, "vt ext f16x8=8");
    // Base template
    CHECK_EQ((vector_traits<float>::vector_size), 1, "vt float=1");
}

// ============================================================
// Test 3: type_convert roundtrip
// ============================================================
__global__ void test_type_convert(float* results) {
    using namespace ck_tile;
    int i = 0;
    // fp8 roundtrip
    float vals[] = {1.0f, -1.0f, 0.5f, 2.0f, 0.0f, 0.125f};
    for (int j = 0; j < 6; j++) {
        fp8_t fp8 = type_convert<fp8_t>(vals[j]);
        float recovered = type_convert<float>(fp8);
        results[i++] = recovered;
        CHECK_NEAR(recovered, vals[j], 0.1f, "fp8 roundtrip");
    }
    // bf16 roundtrip
    for (int j = 0; j < 6; j++) {
        bf16_t bf = type_convert<bf16_t>(vals[j]);
        float recovered = type_convert<float>(bf);
        results[i++] = recovered;
        CHECK_NEAR(recovered, vals[j], 0.01f, "bf16 roundtrip");
    }
    // fp16 roundtrip
    for (int j = 0; j < 6; j++) {
        fp16_t fp = type_convert<fp16_t>(vals[j]);
        float recovered = type_convert<float>(fp);
        results[i++] = recovered;
        CHECK_NEAR(recovered, vals[j], 0.01f, "fp16 roundtrip");
    }
}

// ============================================================
// Test 4: fp8_interpretation enum values (must match CK)
// ============================================================
__global__ void test_fp8_enum() {
    using namespace ck_tile;
    CHECK_EQ(static_cast<int>(fp8_interpretation::E4M3_OCP), 0, "E4M3_OCP=0");
    CHECK_EQ(static_cast<int>(fp8_interpretation::E5M2_OCP), 1, "E5M2_OCP=1");
    CHECK_EQ(static_cast<int>(fp8_interpretation::E4M3_FNUZ), 2, "E4M3_FNUZ=2");
    CHECK_EQ(static_cast<int>(fp8_interpretation::E5M2_FNUZ), 3, "E5M2_FNUZ=3");
}

// ============================================================
// Test 5: numeric limits
// ============================================================
__global__ void test_numeric() {
    using namespace ck_tile;
    CHECK(numeric<float>::max() > 1e30f, "float max > 1e30");
    CHECK(numeric<float>::lowest() < -1e30f, "float lowest < -1e30");
    float fp16_max = type_convert<float>(numeric<fp16_t>::max());
    CHECK_NEAR(fp16_max, 65504.0f, 1.0f, "fp16 max ~65504");
}

// ============================================================
// Test 6: static_for
// ============================================================
__global__ void test_static_for(int* results) {
    using namespace ck_tile;
    int sum = 0;
    static_for<0, 5, 1>{}([&](auto i) {
        sum += i.value;
    });
    results[0] = sum;
    CHECK_EQ(sum, 10, "static_for 0+1+2+3+4=10");
}

// ============================================================
// Test 7: thread_buffer and get_as
// ============================================================
__global__ void test_thread_buffer() {
    using namespace ck_tile;
    thread_buffer<float, 4> buf;
    buf[0] = 1.0f; buf[1] = 2.0f; buf[2] = 3.0f; buf[3] = 4.0f;
    CHECK_NEAR(buf[0], 1.0f, 1e-6f, "tb[0]=1");
    CHECK_NEAR(buf(2), 3.0f, 1e-6f, "tb(2)=3");
    // get_as<int> — reinterpret 16 bytes as 4 ints
    auto& as_int = buf.get_as<int>();
    // float 1.0f = 0x3F800000
    CHECK_EQ(as_int(0), 0x3F800000, "get_as<int>(0)=0x3F800000");
}

// ============================================================
// Test 8: vec_convert (the actual kernel path)
// ============================================================
__global__ void test_vec_convert() {
    using namespace ck_tile;
    thread_buffer<fp8_t, 4> fp8_buf;
    float vals[] = {1.0f, -1.0f, 0.5f, 2.0f};
    for (int i = 0; i < 4; i++)
        fp8_buf[i] = type_convert<fp8_t>(vals[i]);

    auto f32_buf = vec_convert<fp32_t, fp8_t, 4>(fp8_buf);
    for (int i = 0; i < 4; i++)
        CHECK_NEAR(f32_buf[i], vals[i], 0.1f, "vec_convert fp8->f32");
}

int main() {
    float *d_results;
    int *d_int_results;
    int h_pass = 0, h_fail = 0;

    hipMalloc(&d_results, 256 * sizeof(float));
    hipMalloc(&d_int_results, 16 * sizeof(int));

    test_constant_ops<<<1, 1>>>();
    test_vector_traits<<<1, 1>>>();
    test_type_convert<<<1, 1>>>(d_results);
    test_fp8_enum<<<1, 1>>>();
    test_numeric<<<1, 1>>>();
    test_static_for<<<1, 1>>>(d_int_results);
    test_thread_buffer<<<1, 1>>>();
    test_vec_convert<<<1, 1>>>();

    hipDeviceSynchronize();

    hipMemcpyFromSymbol(&h_pass, HIP_SYMBOL(g_pass), sizeof(int));
    hipMemcpyFromSymbol(&h_fail, HIP_SYMBOL(g_fail), sizeof(int));

    printf("ck_tile_shim tests: %d PASS, %d FAIL\n", h_pass, h_fail);
    return h_fail > 0 ? 1 : 0;
}
