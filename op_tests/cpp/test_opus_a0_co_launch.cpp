// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// Native HIP launch test: no PyTorch, rocminfo, GPU RNG or reference device library.
#include "gfx1250/opus_co_launch_gfx1250.cuh"
#if !defined(__HIP_DEVICE_COMPILE__)
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

void check_hip(hipError_t status, const char* operation) {
    if(status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorName(status) +
                                 " / " + hipGetErrorString(status));
}

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}
float from_bf16(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
struct Buffer {
    void* pointer = nullptr;
    explicit Buffer(size_t bytes) { check_hip(hipMalloc(&pointer, bytes), "hipMalloc"); }
    ~Buffer() { if(pointer) (void)hipFree(pointer); }
    Buffer(const Buffer&) = delete;
};
struct Variant {
    int kid, bm, bn, bk, slots, cm, cn, block;
    std::string symbol;
};

void run_case(const Variant& v, int M, int N, int K, int batch, int pad,
              int device, hipStream_t stream) {
    const int sa = K + pad, sb = K + pad;
    std::vector<uint16_t> a(size_t(batch) * M * sa), b(size_t(batch) * N * sb);
    std::vector<uint16_t> output(size_t(batch) * M * N, 0x7fc0);
    for(int z = 0; z < batch; ++z) {
        for(int m = 0; m < M; ++m)
            for(int k = 0; k < sa; ++k)
                a[(size_t(z) * M + m) * sa + k] = to_bf16(float((z * 5 + m * 3 + k * 7) % 9 - 4) * 0.125f);
        for(int n = 0; n < N; ++n)
            for(int k = 0; k < sb; ++k)
                b[(size_t(z) * N + n) * sb + k] = to_bf16(float((z * 3 + n * 5 + k * 2) % 7 - 3) * 0.25f);
    }
    std::cout << "BEGIN kid=" << v.kid << " M/N/K=" << M << '/' << N << '/' << K
              << " batch=" << batch << " pad=" << pad << std::endl;
    Buffer da(a.size() * 2), db(b.size() * 2), dc(output.size() * 2);
    check_hip(hipMemcpy(da.pointer, a.data(), a.size() * 2, hipMemcpyHostToDevice), "copy A");
    check_hip(hipMemcpy(db.pointer, b.data(), b.size() * 2, hipMemcpyHostToDevice), "copy B");
    check_hip(hipMemcpy(dc.pointer, output.data(), output.size() * 2, hipMemcpyHostToDevice), "poison C");

    opus_gemm_4wave_compute_kargs_gfx1250 args{};
    static_assert(sizeof(args) == 64);
    args.ptr_a = da.pointer; args.ptr_b = db.pointer; args.ptr_c = dc.pointer;
    args.stride_a_batch = int64_t(M) * sa; args.stride_b_batch = int64_t(N) * sb;
    args.m = M; args.n = N; args.k = K;
    args.stride_a = sa; args.stride_b = sb; args.stride_c = N;
    const int gm = ((M + v.bm - 1) / v.bm + v.cm - 1) / v.cm * v.cm;
    const int gn = ((N + v.bn - 1) / v.bn + v.cn - 1) / v.cn * v.cn;
    auto& kernel = opus_gfx1250_co::co_kernel(v.symbol, device);
    std::cout << "LOAD_OK kid=" << v.kid << std::endl;
    size_t bytes = sizeof(args);
    for(int replay = 0; replay < 2; ++replay)
        kernel.launch_kernel({&args, &bytes, gm, gn, batch, v.block, 1, 1, stream, v.cm, v.cn, 1});
    std::cout << "LAUNCH_SUBMITTED kid=" << v.kid << std::endl;
    check_hip(hipStreamSynchronize(stream), "CO stream synchronize");
    std::cout << "STREAM_COMPLETE kid=" << v.kid << std::endl;
    check_hip(hipMemcpy(output.data(), dc.pointer, output.size() * 2, hipMemcpyDeviceToHost), "copy C");
    std::cout << "OUTPUT_COPY_COMPLETE kid=" << v.kid << std::endl;
    float max_error = 0;
    for(int z = 0; z < batch; ++z)
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n) {
                float expected = 0;
                for(int k = 0; k < K; ++k)
                    expected += from_bf16(a[(size_t(z) * M + m) * sa + k]) *
                                from_bf16(b[(size_t(z) * N + n) * sb + k]);
                const float actual = from_bf16(output[(size_t(z) * M + m) * N + n]);
                const float error = std::abs(actual - expected);
                if(!std::isfinite(actual) || error > 0.5f + 0.03f * std::abs(expected))
                    throw std::runtime_error("numerical mismatch kid=" + std::to_string(v.kid) +
                        " at batch/row/col=" + std::to_string(z) + '/' + std::to_string(m) + '/' +
                        std::to_string(n) + " got=" + std::to_string(actual) + " expected=" + std::to_string(expected));
                max_error = std::max(max_error, error);
            }
    std::cout << "PASS kid=" << v.kid << " max_abs_error=" << max_error << std::endl;
}

int main(int argc, char** argv) {
    try {
        if(argc < 3) throw std::runtime_error("usage: test <combined CO root> <cases.tsv> [device] [limit]");
        aiter_detail::AiterThrowGuard throwing;
        const int device = argc > 3 ? std::stoi(argv[3]) : 0;
        const int limit = argc > 4 ? std::stoi(argv[4]) : 3;
        check_hip(hipSetDevice(device), "hipSetDevice");
        int runtime = 0;
        check_hip(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
        const auto& info = opus_gfx1250_co::device_info(device);
        std::cout << "HIP_RUNTIME=" << runtime << " DEVICE=" << device << " ARCH=" << info.arch
                  << " ASIC_REVISION=" << info.asic_revision << std::endl;
        if(info.arch != "gfx1250" || info.asic_revision != 0)
            throw std::runtime_error("requested A0 launch test requires current-device gfx1250 ASIC Revision 0");
        setenv("OPUS_GEN_CO_DIR", argv[1], 1);
        std::ifstream file(argv[2]);
        if(!file) throw std::runtime_error("cannot read cases TSV");
        hipStream_t stream = nullptr;
        check_hip(hipStreamCreate(&stream), "hipStreamCreate");
        int count = 0;
        Variant v{};
        while(file >> v.kid >> v.symbol >> v.bm >> v.bn >> v.bk >> v.slots >> v.cm >> v.cn >> v.block) {
            if(limit > 0 && count >= limit) break;
            const auto selected = opus_gfx1250_co::select_image(argv[1], info.arch, info.asic_revision, v.symbol);
            std::cout << "SELECT kid=" << v.kid << " ELF_EXPECTED=0xeb PATH=" << selected.path << std::endl;
            run_case(v, 1, 1, 1, 2, 0, device, stream);
            run_case(v, v.bm + 1, v.bn + 1, v.bk * v.slots + 1, 2, 8, device, stream);
            ++count;
        }
        check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
        std::cout << "A0_NATIVE_CO_LAUNCH_PASS variants=" << count << " cases=" << 2 * count << std::endl;
    } catch(const std::exception& error) {
        std::cerr << "A0_NATIVE_CO_LAUNCH_FAIL: " << error.what() << std::endl;
        return 1;
    }
}
#endif
