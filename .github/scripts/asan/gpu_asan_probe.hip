// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP failure: %s\n", hipGetErrorString(status));
        std::exit(2);
    }
}

__global__ void asan_probe(int* values, int index) {
    values[index] = 42;
}

int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "valid") != 0 &&
                      std::strcmp(argv[1], "invalid") != 0)) {
        std::fprintf(stderr, "Usage: gpu_asan_probe valid|invalid gfx942|gfx950\n");
        return 2;
    }
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count != 1) {
        std::fprintf(stderr, "Expected one assigned GPU, found %d\n", count);
        return 2;
    }
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    std::printf("GPU: %s, arch: %s\n", properties.name, properties.gcnArchName);
    std::fflush(stdout);
    const size_t arch_length = std::strlen(argv[2]);
    if (std::strncmp(properties.gcnArchName, argv[2], arch_length) != 0 ||
        properties.gcnArchName[arch_length] != ':' ||
        std::strstr(properties.gcnArchName, ":xnack+") == nullptr) {
        std::fprintf(stderr, "Expected %s with XNACK enabled\n", argv[2]);
        return 2;
    }

    int* values = nullptr;
    check(hipMalloc(&values, 64 * sizeof(int)));
    // Touch the first red-zone element, not an arbitrary unallocated address.
    const int index = std::strcmp(argv[1], "invalid") == 0 ? 64 : 63;
    hipLaunchKernelGGL(asan_probe, dim3(1), dim3(1), 0, 0, values, index);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    if (index == 63) {
        int result = 0;
        check(hipMemcpy(&result, values + index, sizeof(int), hipMemcpyDeviceToHost));
        if (result != 42) {
            std::fprintf(stderr, "Unexpected kernel result: %d\n", result);
            return 2;
        }
    }
    check(hipFree(values));
    std::puts("Kernel completed");
    return 0;
}
