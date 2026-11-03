// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace opus_gfx1250_co {

struct SelectedImage {
    std::string path;
    int asic_revision;
    uint32_t processor;
};

inline std::string absolute_co_root(const std::string& root) {
    if(root.empty())
        throw std::runtime_error("opus_gemm gfx1250: OPUS_GEN_CO_DIR is empty");
    return std::filesystem::absolute(root).lexically_normal().string();
}

inline SelectedImage select_image(const std::string& root,
                                  const std::string& arch,
                                  int asic_revision,
                                  const std::string& symbol) {
    if(arch != "gfx1250")
        throw std::runtime_error("opus_gemm CO requires gfx1250, running device is " + arch);
    if(asic_revision < 0 || asic_revision > 15)
        throw std::runtime_error("opus_gemm gfx1250: invalid ASIC Revision " +
                                 std::to_string(asic_revision));

    const std::filesystem::path base(absolute_co_root(root));
    const bool a0 = asic_revision == 0;
    const auto is_directory = [](const std::filesystem::path& path) {
        std::error_code error;
        const bool result = std::filesystem::is_directory(path, error);
        if(error && error != std::errc::no_such_file_or_directory &&
           error != std::errc::not_a_directory)
            throw std::runtime_error("opus_gemm: cannot inspect CO directory " +
                                     path.string() + ": " + error.message());
        return result;
    };
    // Combined roots contain gfx1250/ and/or gfx1250-a0/. A direct profile
    // override names the leaf directory itself. Missing A0 files in a combined
    // root stay on the A0 path; never fall back to its regular sibling.
    const bool combined = is_directory(base / arch) || is_directory(base / (arch + "-a0"));
    const auto directory = combined ? base / (a0 ? arch + "-a0" : arch) : base;
    return {(directory / (symbol + ".co")).lexically_normal().string(),
            asic_revision, a0 ? 0xebu : 0x49u};
}

inline std::string co_cache_key(int device, const SelectedImage& image) {
    return std::to_string(device) + ":" + image.path;
}

inline void validate_image_profile(const std::vector<char>& image,
                                   const SelectedImage& selected) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(image[i]); };
    if(image.size() < 64 || byte(0) != 0x7f || byte(1) != 'E' || byte(2) != 'L' ||
       byte(3) != 'F' || byte(4) != 2 || byte(5) != 1)
        throw std::runtime_error("opus_gemm: " + selected.path +
                                 " is not a bare little-endian ELF64 code object");
    const uint32_t machine = byte(18) | (uint32_t(byte(19)) << 8);
    const uint32_t processor = byte(48); // e_flags & EF_AMDGPU_MACH (0xff)
    if(machine != 224 || processor != selected.processor) {
        const std::string profile = selected.asic_revision == 0 ? "A0 gfx1250-strict" : "B0+ gfx1250";
        throw std::runtime_error("opus_gemm: ASIC Revision " +
                                 std::to_string(selected.asic_revision) + " requires " + profile +
                                 "; wrong ELF processor in " + selected.path +
                                 " (got " + std::to_string(processor) + ", expected " +
                                 std::to_string(selected.processor) +
                                 "). Rebuild/select the matching CO profile; no cross-profile fallback.");
    }
}

} // namespace opus_gfx1250_co
