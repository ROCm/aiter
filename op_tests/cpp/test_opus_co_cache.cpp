// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gfx1250/opus_co_launch_gfx1250.cuh"
#if !defined(__HIP_DEVICE_COMPILE__)
#include <cassert>
#include <cstring>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;
namespace test_state {
thread_local int device = 0;
struct Module { int device; const char* image; const void* host = nullptr; };
struct State {
    int created = 0;
    int destroyed = 0;
    bool fail = false;
    std::unordered_map<const void*, Module*> hosts;
};
State& state() { static State* value = new State; return *value; }
}

extern "C" hipError_t hipGetDevice(int* device) {
    *device = test_state::device;
    return hipSuccess;
}
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
    *prop = hipDeviceProp_t{};
    std::strcpy(prop->gcnArchName, "gfx1250");
    prop->asicRevision = device == 1 ? 1 : 0;
    return hipSuccess;
}
extern "C" const char* hipGetErrorString(hipError_t) { return "mock HIP status"; }
extern "C" hipError_t hipGetFuncBySymbol(hipFunction_t* function, const void* host) {
    const auto it = test_state::state().hosts.find(host);
    if(test_state::state().fail || it == test_state::state().hosts.end() ||
       it->second->device != test_state::device) {
        *function = nullptr;
        return hipErrorNoBinaryForGpu;
    }
    *function = reinterpret_cast<hipFunction_t>(it->second);
    return hipSuccess;
}
namespace aiter_detail {
extern "C" void* __hipRegisterFatBinary(const FatBinaryWrapper* wrapper) noexcept {
    ++test_state::state().created;
    return new test_state::Module{test_state::device, static_cast<const char*>(wrapper->binary)};
}
extern "C" void __hipUnregisterFatBinary(void* value) noexcept {
    auto* module = static_cast<test_state::Module*>(value);
    assert(static_cast<unsigned char>(module->image[0]) == 0x7f); // image still alive
    test_state::state().hosts.erase(module->host);
    ++test_state::state().destroyed;
    delete module;
}
extern "C" void __hipRegisterFunction(void* value, const void* host, const char*, const char*,
                                       int, void*, void*, void*, void*, void*) noexcept {
    auto* module = static_cast<test_state::Module*>(value);
    module->host = host;
    test_state::state().hosts[host] = module;
}
}

void write_image(const fs::path& path, unsigned processor) {
    fs::create_directories(path.parent_path());
    std::vector<char> bytes(64, 0);
    bytes[0] = 0x7f; bytes[1] = 'E'; bytes[2] = 'L'; bytes[3] = 'F';
    bytes[4] = 2; bytes[5] = 1; bytes[18] = static_cast<char>(224);
    bytes[48] = static_cast<char>(processor);
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), bytes.size());
}

int main(int argc, char** argv) {
    assert(argc == 2);
    aiter_detail::AiterThrowGuard throwing;
    const auto work = fs::absolute(argv[1]);
    const std::string symbol = "same_kid";
    const auto root = work / "combined";
    const auto alternate = work / "alternate";
    for(const auto& base : {root, alternate}) {
        write_image(base / "gfx1250" / (symbol + ".co"), 0x49);
        write_image(base / "gfx1250-a0" / (symbol + ".co"), 0xeb);
    }
    setenv("OPUS_GEN_CO_DIR", root.c_str(), 1);
    auto* a0 = &opus_gfx1250_co::co_kernel(symbol, 0);
    assert(test_state::state().created == 1);
    assert(&opus_gfx1250_co::co_kernel(symbol, 0) == a0);
    test_state::device = 1;
    auto* b0 = &opus_gfx1250_co::co_kernel(symbol, 1);
    assert(b0 != a0 && test_state::state().created == 2);
    test_state::device = 0;
    assert(&opus_gfx1250_co::co_kernel(symbol, 0) == a0);
    setenv("OPUS_GEN_CO_DIR", (root / "gfx1250-a0").c_str(), 1);
    assert(&opus_gfx1250_co::co_kernel(symbol, 0) == a0);
    setenv("OPUS_GEN_CO_DIR", alternate.c_str(), 1);
    auto* other = &opus_gfx1250_co::co_kernel(symbol, 0);
    assert(other != a0 && test_state::state().created == 3);

    const auto retry_root = work / "retry";
    write_image(retry_root / "gfx1250-a0" / (symbol + ".co"), 0xeb);
    setenv("OPUS_GEN_CO_DIR", retry_root.c_str(), 1);
    test_state::state().fail = true;
    bool failed = false;
    try { opus_gfx1250_co::co_kernel(symbol, 0); }
    catch(const std::runtime_error&) { failed = true; }
    assert(failed && test_state::state().created == 4 && test_state::state().destroyed == 1);
    test_state::state().fail = false;
    auto* retried = &opus_gfx1250_co::co_kernel(symbol, 0);
    assert(retried != other && test_state::state().created == 5);
    assert(&opus_gfx1250_co::co_kernel(symbol, 0) == retried);

    const auto bad = work / "wrong-profile";
    write_image(bad / "gfx1250" / (symbol + ".co"), 0x49);
    setenv("OPUS_GEN_CO_DIR", bad.c_str(), 1);
    failed = false;
    try { opus_gfx1250_co::co_kernel(symbol, 0); }
    catch(const std::runtime_error&) { failed = true; }
    assert(failed && test_state::state().created == 5); // rejected before registration
    setenv("OPUS_GEN_CO_DIR", root.c_str(), 1);
    assert(&opus_gfx1250_co::co_kernel(symbol, 0) == a0);
    failed = false;
    try { opus_gfx1250_co::co_kernel(symbol, 1); }
    catch(const std::runtime_error&) { failed = true; }
    assert(failed);

    // No environment mutation while threads are running. Each keeps its own
    // current device and TLS fast path; shared modules remain process-lifetime.
    std::thread ta([&] { test_state::device = 0; assert(&opus_gfx1250_co::co_kernel(symbol, 0) == a0); });
    std::thread tb([&] { test_state::device = 1; assert(&opus_gfx1250_co::co_kernel(symbol, 1) == b0); });
    ta.join(); tb.join();
    assert(test_state::state().created == 5);
    std::cout << "CO_DEVICE_ROOT_CACHE_AND_REGISTRATION_ROLLBACK_TESTS_PASS\n";
}
#endif
