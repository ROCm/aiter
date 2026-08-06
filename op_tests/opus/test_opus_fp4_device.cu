// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_opus_fp4_device.cpp
 * @brief On-device (GPU) tests for the sub-byte fp4_t packing refactor.
 *
 * fp4_t is now ONE logical 4-bit element (cutlass float_e2m1_t style); opus::array
 * bit-packs N values into ceil(N*4/8) bytes and exposes a proxy reference. The real
 * fp4<->fp32 conversion intrinsics only exist on gfx950 (MI350/MI355), so this test
 * runs the round-trip and packing checks on the device.
 *
 * Standalone HIP executable (no torch / rocprim). If no GPU is visible it prints
 * SKIP and exits 0, so it is safe to run in CI on CPU-only hosts.
 *
 * Built by build.sh; hipcc targets the machine's default gfx arch automatically.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include "opus/opus.hpp"

using namespace opus;

#define HIP_CHECK(expr)                                                              \
    do {                                                                             \
        hipError_t _e = (expr);                                                      \
        if (_e != hipSuccess) {                                                      \
            printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
            return false;                                                            \
        }                                                                            \
    } while (0)

// ── Device kernels ─────────────────────────────────────────────────────────

// fp32x8 -> fp4 (packed) -> fp32x8 round trip via opus::cast (real gfx950 intrinsics)
__global__ void fp4_roundtrip_kernel(const float* in, float* out) {
    fp32x8_t v;
    for (int i = 0; i < 8; ++i) v[i] = in[i];
    auto q = opus::cast<fp4_t>(v);        // array<fp4_t,8> : 8 values in 4 bytes
    auto d = opus::cast<fp32_t>(q);       // back to fp32x8
    for (int i = 0; i < 8; ++i) out[i] = d[i];
}

// Build a packed array<fp4_t,8> from codes 0..7 on-device; emit the 4 raw bytes.
// Verifies the device-side proxy write + nibble layout matches the host.
__global__ void fp4_pack_layout_kernel(unsigned char* out_bytes) {
    array<fp4_t, 8> a;
    for (int i = 0; i < 8; ++i) { fp4_t v; v.value = (unsigned char)i; a[i] = v; }
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&a);
    for (int i = 0; i < 4; ++i) out_bytes[i] = p[i];
}

// ── Host driver ────────────────────────────────────────────────────────────

static bool test_fp4_roundtrip() {
    // All values exactly representable in e2m1 (sign, 2 exp, 1 mantissa): 0,.5,1,1.5,2,3,4,6
    float h_in[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(h_in)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(h_in)));
    HIP_CHECK(hipMemcpy(d_in, h_in, sizeof(h_in), hipMemcpyHostToDevice));
    fp4_roundtrip_kernel<<<1, 1>>>(d_in, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    float h_out[8];
    HIP_CHECK(hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (h_out[i] != h_in[i]) {
            printf("  fp4 roundtrip mismatch [%d] in=%g out=%g\n", i, h_in[i], h_out[i]);
            ok = false;
        }
    }
    return ok;
}

static bool test_fp4_pack_layout() {
    unsigned char* d_bytes;
    HIP_CHECK(hipMalloc(&d_bytes, 4));
    fp4_pack_layout_kernel<<<1, 1>>>(d_bytes);
    HIP_CHECK(hipDeviceSynchronize());
    unsigned char h[4];
    HIP_CHECK(hipMemcpy(h, d_bytes, 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_bytes));
    const unsigned char want[4] = {0x10, 0x32, 0x54, 0x76};   // elem i -> byte i/2, nibble i%2
    bool ok = std::memcmp(h, want, 4) == 0;
    if (!ok)
        printf("  pack layout got %02x %02x %02x %02x want 10 32 54 76\n", h[0], h[1], h[2], h[3]);
    return ok;
}

int main() {
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        printf("SKIP: no GPU visible (fp4 device tests require gfx950/MI350+)\n");
        return 0;
    }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("======================================\n");
    printf("OPUS fp4 device tests (arch=%s)\n", prop.gcnArchName);
    printf("======================================\n");
    if (std::strstr(prop.gcnArchName, "gfx950") == nullptr) {
        printf("SKIP: fp4 conversion intrinsics require gfx950 (found %s)\n", prop.gcnArchName);
        return 0;
    }

    int passed = 0, failed = 0;
    struct { const char* name; bool (*fn)(); } tests[] = {
        {"test_fp4_roundtrip", test_fp4_roundtrip},
        {"test_fp4_pack_layout", test_fp4_pack_layout},
    };
    for (auto& t : tests) {
        printf("Running %s... ", t.name);
        fflush(stdout);
        if (t.fn()) { printf("PASSED\n"); passed++; }
        else        { printf("FAILED\n"); failed++; }
    }
    printf("======================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("======================================\n");
    return failed > 0 ? 1 : 0;
}
