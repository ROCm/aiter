# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile the PA v1 template's workspace offsets on the host, without HIP.

This checks separation of the statistics and temporary-output buffers only.
GPU attention correctness, including MTP writer/reducer indexing, is separate.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from jinja2 import Template


@pytest.mark.parametrize("mtp", [1, 2, 4])
def test_workspace_regions_do_not_overlap(tmp_path, mtp):
    # Both supported query types (FP16/BF16) occupy two bytes.
    dtype = "uint16_t"
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    template = Path(__file__).resolve().parents[1] / "csrc/cpp_itfs/pa/pa_v1.cpp.jinja"
    source = Template(template.read_text()).render(
        mtp=mtp, dtype=dtype, logits_soft_cap_enabled=False
    )
    # Compile the actual pointer declarations, rather than a copied formula.
    offsets = source.split("const float logits_soft_cap_rcp = 0.f;", 1)[1].split(
        "ck_tile::ComposedAttention", 1
    )[0]
    harness = tmp_path / "workspace.cpp"
    harness.write_text("""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
int main() {
    for (int num_seqs : {1, 4, 17}) {
        for (int num_heads : {8, 16, 64}) {
            for (int max_num_partitions : {1, 2, 33}) {
                const int head_size = 128;
                const int64_t count =
                    int64_t(num_seqs) * MTP * num_heads * max_num_partitions;
                const int64_t stats_bytes = count * sizeof(float);
                const int64_t output_bytes = count * head_size * sizeof(SCALAR);
                const int64_t total_bytes = 2 * stats_bytes + output_bytes;
                std::vector<float> workspace(total_bytes / sizeof(float) + 4);
                auto* workspace_buffer_ptr = workspace.data();
                auto* bytes = reinterpret_cast<unsigned char*>(workspace_buffer_ptr);
                std::fill(bytes, bytes + total_bytes + 16, 0xDE);
                OFFSETS
                assert(reinterpret_cast<unsigned char*>(max_logits_ptr) - bytes
                       == stats_bytes);
                assert(reinterpret_cast<unsigned char*>(tmp_out_ptr) - bytes
                       == 2 * stats_bytes);
                std::fill_n(reinterpret_cast<unsigned char*>(exp_sums_ptr),
                            stats_bytes, 0x11);
                std::fill_n(reinterpret_cast<unsigned char*>(max_logits_ptr),
                            stats_bytes, 0x22);
                std::fill_n(reinterpret_cast<unsigned char*>(tmp_out_ptr),
                            output_bytes, 0x33);
                for (int64_t i = 0; i < total_bytes + 16; ++i) {
                    const auto expected = i < stats_bytes ? 0x11
                        : i < 2 * stats_bytes ? 0x22
                        : i < total_bytes ? 0x33 : 0xDE;
                    assert(bytes[i] == expected);
                }
            }
        }
    }
}
""".replace("MTP", str(mtp)).replace("SCALAR", dtype).replace("OFFSETS", offsets))
    binary = tmp_path / "workspace"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
