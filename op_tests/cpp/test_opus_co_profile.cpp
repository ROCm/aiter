// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gfx1250/opus_co_profile_gfx1250.h"
#include <cassert>
#include <iostream>

using namespace opus_gfx1250_co;
namespace fs = std::filesystem;

std::vector<char> elf(uint32_t processor) {
    std::vector<char> bytes(64, 0);
    bytes[0] = 0x7f; bytes[1] = 'E'; bytes[2] = 'L'; bytes[3] = 'F';
    bytes[4] = 2; bytes[5] = 1;
    bytes[18] = static_cast<char>(224);
    bytes[48] = static_cast<char>(processor);
    bytes[49] = 2; // Higher feature bits must not affect the processor test.
    return bytes;
}

template<class F> void must_throw(F f) {
    bool threw = false;
    try { f(); } catch(const std::runtime_error&) { threw = true; }
    assert(threw);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const auto work = fs::absolute(argv[1]).lexically_normal();
    fs::create_directories(work / "combined" / "gfx1250-a0");
    fs::create_directories(work / "combined" / "gfx1250");
    fs::create_directories(work / "flat-a0");
    const auto base = work / "combined";
    const std::string symbol = "kid_496_same_symbol";
    const auto a0 = select_image(base.string(), "gfx1250", 0, symbol);
    const auto b0 = select_image(base.string(), "gfx1250", 1, symbol);
    assert(a0.path == (base / "gfx1250-a0" / (symbol + ".co")).string());
    assert(b0.path == (base / "gfx1250" / (symbol + ".co")).string());
    assert(fs::path(a0.path).filename() == fs::path(b0.path).filename());
    assert(a0.processor == 0xeb && b0.processor == 0x49);
    validate_image_profile(elf(0xeb), a0);
    validate_image_profile(elf(0x49), b0);
    must_throw([&] { validate_image_profile(elf(0x49), a0); });
    must_throw([&] { validate_image_profile(elf(0xeb), b0); });
    must_throw([&] { validate_image_profile(std::vector<char>(4), a0); });
    must_throw([&] { select_image(base.string(), "gfx950", 0, symbol); });
    must_throw([&] { select_image(base.string(), "gfx1250", -1, symbol); });
    must_throw([&] { select_image(base.string(), "gfx1250", 16, symbol); });
    const auto c0 = select_image(base.string(), "gfx1250", 2, symbol);
    assert(c0.path == b0.path && c0.processor == 0x49);

    // Direct-profile overrides never append a0 twice.
    assert(select_image((base / "gfx1250-a0").string(), "gfx1250", 0, symbol).path == a0.path);
    const auto flat = select_image((work / "flat-a0").string(), "gfx1250", 0, symbol);
    assert(flat.path == (work / "flat-a0" / (symbol + ".co")).string());
    validate_image_profile(elf(0xeb), flat);
    // No file was created: an existing combined A0 subtree still selects the
    // missing A0 filename, never the regular image.
    assert(!fs::exists(a0.path));
    assert(a0.path != b0.path);

    // HIP visibility can reorder device IDs. Stepping is supplied per device,
    // and cache entries isolate both ordinal and exact selected image.
    assert(co_cache_key(0, a0) != co_cache_key(1, a0));
    assert(co_cache_key(0, a0) != co_cache_key(0, b0));
    assert(co_cache_key(0, b0) == co_cache_key(0, c0));
    const auto old_cwd = fs::current_path();
    fs::current_path(work);
    const auto relative = select_image("combined", "gfx1250", 0, symbol);
    assert(relative.path == a0.path);
    fs::current_path(base);
    assert(absolute_co_root(".") != absolute_co_root("../"));
    fs::current_path(old_cwd);
    std::cout << "CO_PROFILE_CPP17_TESTS_PASS\n";
}
