# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate Opus MoE dispatch headers and device-instantiation shards.

This is intentionally smaller than ``csrc/opus_gemm/gen_instances.py`` today:
several kernels still live in hand-written headers, but the generated
manifests are the single source of truth for kid -> launcher mapping.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from opus_moe_common import (
    OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT,
    OPUS_A8W4_ROUTE_REDUCE_INSTANCES,
    OPUS_A8W4_STAGE1_MFMA_K,
    OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K,
    STAGE1_A8W4_KERNELS,
    STAGE2_A8W4_KERNELS,
    STAGE2_BF16_KERNELS,
    opus_a8w4_stage1_shape_requirements,
)

BF16_MANIFEST_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// BF16 stage2 kid -> launcher manifest. This is deliberately generated from
// opus_moe_common.py so Python tuner metadata and C++ dispatch tables do not
// drift as more stage2 kids land.

"""

A8W4_MANIFEST_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// A8W4 stage2 decode kid -> launcher cases. Generated from structured
// metadata so Python tuner metadata and C++ dispatch cases do not drift.

"""

A8W4_META_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// A8W4 stage2 decode metadata generated from
// aiter/ops/opus/moe_stage2_a8w4_meta.py.

namespace opus_moe
{

"""

A8W4_META_FOOTER = """
} // namespace opus_moe
"""

STAGE1_A8W4_META_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// A8W4 stage1 metadata generated from
// csrc/opus_moe/opus_moe_common.py.

namespace opus_moe
{

"""

STAGE1_A8W4_MANIFEST_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// A8W4 stage1 kid -> launcher cases. Generated from structured metadata so
// csrc metadata and C++ dispatch cases do not drift.

"""


# ---- BF16 private manifest -------------------------------------------------


def _emit_bf16_manifest_header() -> str:
    lines = [BF16_MANIFEST_HEADER]
    bf16_kernels = [STAGE2_BF16_KERNELS[kid] for kid in sorted(STAGE2_BF16_KERNELS)]

    if not bf16_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_DISPATCH_CASES\n\n")
    else:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_DISPATCH_CASES \\\n")
        for idx, inst in enumerate(bf16_kernels):
            suffix = " \\\n" if idx != len(bf16_kernels) - 1 else "\n"
            lines.append(
                f"    case {inst.kid}: return &{inst.launcher}<{inst.trait}>;" + suffix
            )
    lines.append("\n")

    return "".join(lines)


# ---- Shared C++ emit helpers ----------------------------------------------


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _cpp_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _cpp_name_suffix(name: str) -> str:
    return "".join(
        part[:1].upper() + part[1:]
        for part in str(name).replace("-", "_").split("_")
        if part
    )


def _stage2_a8w4_traits_alias(kid: int) -> str:
    return f"OpusMoeStage2A8W4DecodeKid{int(kid)}Traits"


def _stage2_a8w4_traits_type(inst) -> str:
    return (
        "OpusMoeStage2A8W4DecodeShape<"
        f"{inst.block_m}, "
        f"{inst.block_n}, "
        f"{_cpp_bool(inst.direct_atomic)}, "
        f"{_cpp_bool(inst.pace_route_blocks_to_pow2)}, "
        f"{inst.block_threads}, "
        f"{inst.min_blocks_per_cu}, "
        f"{inst.cachectl_b}, "
        f"{inst.cachectl_wscale}, "
        f"{inst.pair_slots}, "
        f"{inst.steady_pair_slots}"
        ">"
    )


def _stage1_a8w4_traits_alias(kid: int) -> str:
    return f"OpusMoeStage1A8W4Kid{int(kid)}Traits"


# ---- A8W4 stage2 metadata and dispatch manifests ---------------------------


def _emit_stage2_a8w4_meta_header() -> str:
    lines = [A8W4_META_HEADER]
    k = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    a8w4_kernels = [STAGE2_A8W4_KERNELS[kid] for kid in sorted(STAGE2_A8W4_KERNELS)]
    block_ms = sorted({inst.block_m for inst in a8w4_kernels})
    block_ns = sorted({inst.block_n for inst in a8w4_kernels})

    for block_m in block_ms:
        lines.append(f"constexpr int kStage2A8W4DecodeBlockM{block_m} = {block_m};\n")
    for block_n in block_ns:
        lines.append(f"constexpr int kStage2A8W4DecodeBlockN{block_n} = {block_n};\n")
    lines.extend(
        [
            (
                "constexpr int kStage2A8W4DecodeDefaultBlockM = "
                f"kStage2A8W4DecodeBlockM{k.default_block_m};\n"
            ),
            (
                "constexpr int kStage2A8W4DecodeDefaultBlockN = "
                f"kStage2A8W4DecodeBlockN{k.default_block_n};\n"
            ),
            f"constexpr int kStage2A8W4DecodeDefaultCtaThreads = {k.default_cta_threads};\n",
            f"constexpr int kStage2A8W4DecodeBKLogical = {k.bk_logical};\n",
            f"constexpr int kStage2A8W4DecodeMfmaM = {k.mfma_m};\n",
            f"constexpr int kStage2A8W4DecodeMfmaN = {k.mfma_n};\n",
            f"constexpr int kStage2A8W4DecodeMfmaK = {k.mfma_k};\n",
            f"constexpr int kStage2A8W4DecodeFp4ValuesPerByte = {k.fp4_values_per_byte};\n",
            f"constexpr int kStage2A8W4DecodeVectorBytes = {k.vector_bytes};\n",
            (
                "constexpr int kStage2A8W4DecodeScaleGroupsPerRowPack = "
                f"{k.scale_groups_per_row_pack};\n"
            ),
            (
                "constexpr int kStage2A8W4DecodeScaleWordsPerGroupPack = "
                f"{k.scale_words_per_group_pack};\n"
            ),
            f"constexpr int kStage2A8W4DecodeCVec = {k.c_vec};\n",
            f"constexpr int kStage2A8W4DecodeCValuesPerAtomic = {k.c_values_per_atomic};\n\n",
        ]
    )

    for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES:
        suffix = _cpp_name_suffix(inst.name)
        lines.extend(
            [
                (
                    f"constexpr int kStage2A8W4RouteReduce{suffix}BlockN = "
                    f"{inst.block_n};\n"
                ),
                (
                    f"constexpr int kStage2A8W4RouteReduce{suffix}Threads = "
                    f"{inst.threads};\n"
                ),
            ]
        )
    lines.append(
        "\n#define GENERATE_OPUS_MOE_STAGE2_A8W4_ROUTE_REDUCE_DISPATCH_CASES(TOPK) \\\n"
    )
    for idx, inst in enumerate(OPUS_A8W4_ROUTE_REDUCE_INSTANCES):
        suffix = _cpp_name_suffix(inst.name)
        line_suffix = (
            " \\\n" if idx != len(OPUS_A8W4_ROUTE_REDUCE_INSTANCES) - 1 else "\n"
        )
        lines.append(
            f"    case opus_moe::kStage2A8W4RouteReduce{suffix}BlockN: "
            "opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<"
            f"opus_moe::kStage2A8W4RouteReduce{suffix}BlockN, "
            f"opus_moe::kStage2A8W4RouteReduce{suffix}Threads, "
            "TOPK>(kargs, grid, stream); break;" + line_suffix
        )
    lines.append("\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_sort_block_m(int kid)\n"
        "{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {inst.sort_block_m};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr bool stage2_a8w4_kid_is_valid(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}:\n")
    lines.append("        return true;\n    default: return false;\n    }\n}\n\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_block_n(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {inst.block_n};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr bool stage2_a8w4_effective_inter_dim_is_supported(\n"
        "    int effective_inter_dim)\n"
        "{\n"
        "    constexpr int kStepPacked =\n"
        "        kStage2A8W4DecodeBKLogical / kStage2A8W4DecodeFp4ValuesPerByte;\n"
        "    return effective_inter_dim >= 2 * kStepPacked &&\n"
        "           effective_inter_dim % kStepPacked == 0;\n"
        "}\n\n"
    )

    lines.append(
        "constexpr bool stage2_a8w4_kid_uses_route_out(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {_cpp_bool(inst.route_out)};\n")
    lines.append("    default: return false;\n    }\n}\n\n")

    lines.append(
        "constexpr bool stage2_a8w4_kid_route_fp8(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {_cpp_bool(inst.route_out_fp8)};\n")
    lines.append("    default: return false;\n    }\n}\n\n")

    lines.append(
        "constexpr const char* stage2_a8w4_kid_name(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f'    case {inst.kid}: return "{_cpp_string(inst.name)}";\n')
    lines.append('    default: return "unknown";\n    }\n}\n\n')

    lines.append(
        "constexpr int stage2_a8w4_auto_direct_atomic_kid(int block_m)\n{\n"
        "    switch(block_m)\n"
        "    {\n"
    )
    for inst in a8w4_kernels:
        if inst.direct_atomic and inst.mode_default:
            lines.append(f"    case {inst.sort_block_m}: return {inst.kid};\n")
    lines.append("    default: return -1;\n    }\n}\n")
    lines.append(A8W4_META_FOOTER)
    return "".join(lines)


def _emit_stage2_a8w4_manifest_header() -> str:
    lines = [A8W4_MANIFEST_HEADER]
    a8w4_kernels = [STAGE2_A8W4_KERNELS[kid] for kid in sorted(STAGE2_A8W4_KERNELS)]

    if not a8w4_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES\n")
        return "".join(lines)

    lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES \\\n")
    for idx, inst in enumerate(a8w4_kernels):
        suffix = " \\\n" if idx != len(a8w4_kernels) - 1 else "\n"
        lines.append(
            f"    case {inst.kid}: "
            "return opus_moe_stage2_a8w4_decode_launch_gfx950<"
            f"{_stage2_a8w4_traits_type(inst)}>(kargs, stream);" + suffix
        )
    lines.append("\n")
    return "".join(lines)


def _stage1_cpp_shape(inst) -> str:
    policy_args = (
        inst.gate_up_group_split,
        inst.k_wave,
        inst.min_blocks_per_cu_override,
        inst.quant_group_blocks,
        inst.block_threads,
        inst.weight_load_stream,
        inst.weight_load_reverse,
        inst.sparse_epilogue,
        inst.k_loop_swizzle_colors,
        inst.route_affinity_window,
        inst.route_affinity_phase_period,
        inst.m_fragment_major,
        inst.b_k1_lead,
    )
    policy = (
        "OpusMoeStage1A8W4Policy<"
        + ", ".join(
            _cpp_bool(arg) if isinstance(arg, bool) else str(arg) for arg in policy_args
        )
        + ">"
    )
    return f"OpusMoeStage1A8W4Shape<" f"{inst.block_m}, {inst.block_n}, {policy}>"


def _emit_stage1_a8w4_meta_header() -> str:
    lines = [STAGE1_A8W4_META_HEADER]
    kernels = [STAGE1_A8W4_KERNELS[kid] for kid in sorted(STAGE1_A8W4_KERNELS)]
    scale_group = OPUS_A8W4_STAGE1_SCALE_GROUP_LOGICAL_K

    lines.extend(
        [
            (
                "constexpr int kOpusMoeStage1A8W4ScaleGroupLogicalK = "
                f"{scale_group};\n\n"
            ),
            "constexpr int kStage1A8W4KidInvalid = -1;\n\n",
        ]
    )

    lines.append(
        "inline bool stage1_a8w4_name_eq(const char* lhs, const char* rhs)\n"
        "{\n"
        "    if(lhs == nullptr || rhs == nullptr)\n"
        "        return false;\n"
        "    while(*lhs != '\\0' && *rhs != '\\0' && *lhs == *rhs)\n"
        "    {\n"
        "        ++lhs;\n"
        "        ++rhs;\n"
        "    }\n"
        "    return *lhs == *rhs;\n"
        "}\n\n"
        "inline int stage1_a8w4_kid_from_name(const char* name)\n"
        "{\n"
    )
    for inst in kernels:
        names = [inst.name]
        if inst.profile_name != inst.name:
            names.append(inst.profile_name)
        condition = " || ".join(
            f'stage1_a8w4_name_eq(name, "{_cpp_string(name)}")' for name in names
        )
        lines.append(f"    if({condition})\n        return {inst.kid};\n")
    lines.append("    return kStage1A8W4KidInvalid;\n}\n")
    lines.append("\n")

    lines.append(
        "constexpr int stage1_a8w4_kid_sort_block_m(int kid)\n"
        "{\n    switch(kid)\n    {\n"
    )
    for inst in kernels:
        lines.append(f"    case {inst.kid}: " f"return {inst.block_m};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.extend(
        [
            "constexpr bool stage1_a8w4_kid_supports_shape(\n",
            "    int kid,\n",
            "    int model_dim,\n",
            "    int logical_inter_dim,\n",
            "    int inter_dim_pad)\n",
            "{\n",
            f"    constexpr int kMfmaK = {OPUS_A8W4_STAGE1_MFMA_K};\n",
            f"    constexpr int kScaleGroup = {scale_group};\n",
            "    const int effective_inter_dim = logical_inter_dim - inter_dim_pad;\n",
            "    if(inter_dim_pad < 0 || effective_inter_dim <= 0)\n",
            "        return false;\n",
            "    if(model_dim % kMfmaK != 0 ||\n",
            "       logical_inter_dim % kScaleGroup != 0)\n",
            "        return false;\n",
            "    const int k_steps = model_dim / kMfmaK;\n",
            "    switch(kid)\n",
            "    {\n",
        ]
    )
    for inst in kernels:
        requirements = opus_a8w4_stage1_shape_requirements(inst)
        if requirements is None:
            raise ValueError(
                f"Opus A8W4 stage1 kid {inst.kid} has invalid shape requirements"
            )
        k_step_multiple, output_cols = requirements
        lines.append(
            f"    case {inst.kid}: "
            f"return k_steps % {k_step_multiple} == 0 && "
            f"effective_inter_dim % {output_cols} == 0;\n"
        )
    lines.append("    default: return false;\n    }\n")
    lines.append("}\n\n")

    lines.append(
        "constexpr const char* stage1_a8w4_kid_name(int kid)\n"
        "{\n    switch(kid)\n    {\n"
    )
    for inst in kernels:
        lines.append(f"    case {inst.kid}: " f'return "{_cpp_string(inst.name)}";\n')
    lines.append('    default: return "unknown";\n    }\n}\n')
    lines.append(A8W4_META_FOOTER)
    return "".join(lines)


def _emit_stage1_a8w4_manifest_header() -> str:
    lines = [STAGE1_A8W4_MANIFEST_HEADER]
    kernels = [STAGE1_A8W4_KERNELS[kid] for kid in sorted(STAGE1_A8W4_KERNELS)]

    if not kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE1_A8W4_DISPATCH_CASES\n")
        return "".join(lines)

    lines.append("#define GENERATE_OPUS_MOE_STAGE1_A8W4_DISPATCH_CASES \\\n")
    for idx, inst in enumerate(kernels):
        suffix = " \\\n" if idx != len(kernels) - 1 else "\n"
        lines.append(
            f"    case {inst.kid}: "
            f"return launch_gfx950<{_stage1_cpp_shape(inst)}>("
            "effective_inter_dim, sorted_blocks, kargs, stream);" + suffix
        )
    lines.append("\n")
    return "".join(lines)


# ---- Device translation-unit generation -----------------------------------


def _route_reduce_instantiation_rows() -> list[tuple[int, int]]:
    rows = [
        (2048, 256),
        (4096, 256),
        *(
            (int(inst.block_n), int(inst.threads))
            for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES
        ),
    ]
    return list(dict.fromkeys(rows))


class OpusMoeDeviceCodegen:
    """Emit independently compilable device-instantiation shards."""

    def __init__(self, working_path: Path):
        self.instances_path = working_path / "instances"

    def _prepare_dirs(self) -> None:
        if self.instances_path.exists():
            shutil.rmtree(self.instances_path)
        self.instances_path.mkdir(parents=True, exist_ok=True)

    def _emit_stage2_device_tus(self) -> None:
        lines = [
            "// SPDX-License-Identifier: MIT\n",
            "// Auto-generated runtime-K A8W4 stage2 device shard; do not edit.\n",
            '#include "gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh"\n',
        ]
        seen_traits: set[str] = set()
        for inst in (STAGE2_A8W4_KERNELS[kid] for kid in sorted(STAGE2_A8W4_KERNELS)):
            traits_type = _stage2_a8w4_traits_type(inst)
            if traits_type in seen_traits:
                continue
            seen_traits.add(traits_type)
            alias = _stage2_a8w4_traits_alias(inst.kid)
            lines.extend(
                [
                    f"using {alias} = {traits_type};\n",
                    "template __global__ void opus_moe_stage2_a8w4_decode_kernel_gfx950<",
                    f"{alias}>(opus_moe_stage2_a8w4_kargs);\n",
                ]
            )
        (self.instances_path / "opus_moe_stage2_a8w4.device.cu").write_text(
            "".join(lines), encoding="utf-8"
        )

    def _emit_stage1_device_tus(self) -> None:
        families = (
            (
                True,
                "group_split",
                "opus_moe_stage1_a8w4_pipeline_group_split_gfx950.cuh",
                "opus_moe_stage1_a8w4_kernel_group_split_gfx950",
            ),
            (
                False,
                "pair_kwave",
                "opus_moe_stage1_a8w4_pipeline_pair_kwave_gfx950.cuh",
                "opus_moe_stage1_a8w4_kernel_pair_kwave_gfx950",
            ),
        )
        for group_split, namespace, header, kernel in families:
            instances = [
                STAGE1_A8W4_KERNELS[kid]
                for kid in sorted(STAGE1_A8W4_KERNELS)
                if bool(STAGE1_A8W4_KERNELS[kid].gate_up_group_split) == group_split
            ]
            if not instances:
                continue
            lines = [
                "// SPDX-License-Identifier: MIT\n",
                "// Auto-generated A8W4 stage1 device shard; do not edit.\n",
                f'#include "gfx950/a8w4/stage1/{header}"\n',
                "namespace opus_moe\n{\nnamespace stage1_a8w4\n{\n",
            ]
            for inst in instances:
                lines.append(
                    f"using {_stage1_a8w4_traits_alias(inst.kid)} = {_stage1_cpp_shape(inst)};\n"
                )
            lines.append(f"namespace pipeline_{namespace}\n{{\n")
            for inst in instances:
                alias = _stage1_a8w4_traits_alias(inst.kid)
                lines.extend(
                    [
                        f"template __global__ void {kernel}<",
                        f"{alias}>(OpusMoeStage1A8W4Kargs);\n",
                    ]
                )
            lines.extend(
                [
                    f"}} // namespace pipeline_{namespace}\n",
                    "} // namespace stage1_a8w4\n",
                    "} // namespace opus_moe\n",
                ]
            )
            path = self.instances_path / f"opus_moe_stage1_a8w4_{namespace}.device.cu"
            path.write_text("".join(lines), encoding="utf-8")

    def _emit_route_reduce_device_tus(self) -> None:
        for topk in (0, 4, 6, 8):
            lines = [
                "// SPDX-License-Identifier: MIT\n",
                "// Auto-generated route-reduce device shard; do not edit.\n",
                "#define OPUS_MOE_ROUTE_REDUCE_DEVICE_TU 1\n",
                '#include "gfx950/opus_moe_stage2_route_output_reduce_gfx950.cuh"\n',
            ]
            for block_n, threads in _route_reduce_instantiation_rows():
                for route_fp8 in (False, True):
                    lines.extend(
                        [
                            "template __global__ void ",
                            "opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950<",
                            f"{block_n}, {threads}, {topk}, {_cpp_bool(route_fp8)}>(",
                            "opus_moe_stage2_route_reduce_kargs);\n",
                        ]
                    )
            path = self.instances_path / f"opus_moe_route_reduce_topk_{topk}.device.cu"
            path.write_text("".join(lines), encoding="utf-8")

    def _emit_bf16_device_tu(self) -> None:
        lines = [
            "// SPDX-License-Identifier: MIT\n",
            "// Auto-generated BF16 stage2 device shard; do not edit.\n",
            '#include "gfx950/a16w16/opus_moe_pipeline_stage2_gemmstyle_gfx950.cuh"\n',
        ]
        for inst in (STAGE2_BF16_KERNELS[kid] for kid in sorted(STAGE2_BF16_KERNELS)):
            lines.extend(
                [
                    "template __global__ void opus_moe_stage2_gemmstyle_kernel_gfx950<",
                    f"{inst.trait}>(opus_moe_stage2_bf16_kargs);\n",
                ]
            )
        (self.instances_path / "opus_moe_stage2_bf16.device.cu").write_text(
            "".join(lines), encoding="utf-8"
        )

    def gen_instances(self) -> None:
        self._prepare_dirs()
        self._emit_stage2_device_tus()
        self._emit_stage1_device_tus()
        self._emit_route_reduce_device_tus()
        self._emit_bf16_device_tu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Opus MoE dispatch headers")
    parser.add_argument("--working_path", required=True)
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_manifest_path = out_dir / "opus_moe_stage2_manifest.h"
    bf16_manifest_path.write_text(_emit_bf16_manifest_header(), encoding="utf-8")
    stage2_a8w4_meta_path = out_dir / "opus_moe_stage2_a8w4_meta.h"
    stage2_a8w4_meta_path.write_text(_emit_stage2_a8w4_meta_header(), encoding="utf-8")
    stage2_a8w4_manifest_path = out_dir / "opus_moe_stage2_a8w4_manifest.h"
    stage2_a8w4_manifest_path.write_text(
        _emit_stage2_a8w4_manifest_header(), encoding="utf-8"
    )
    stage1_a8w4_meta_path = out_dir / "opus_moe_stage1_a8w4_meta.h"
    stage1_a8w4_meta_path.write_text(_emit_stage1_a8w4_meta_header(), encoding="utf-8")
    stage1_a8w4_manifest_path = out_dir / "opus_moe_stage1_a8w4_manifest.h"
    stage1_a8w4_manifest_path.write_text(
        _emit_stage1_a8w4_manifest_header(), encoding="utf-8"
    )
    OpusMoeDeviceCodegen(out_dir).gen_instances()

    print(
        f"[opus_moe gen_instances] wrote {bf16_manifest_path} with "
        f"{len(STAGE2_BF16_KERNELS)} BF16 stage2 kid(s)"
    )
    print(
        f"[opus_moe gen_instances] wrote {stage2_a8w4_manifest_path} with "
        f"{len(STAGE2_A8W4_KERNELS)} runtime-K A8W4 stage2 kid(s)"
    )
    print(f"[opus_moe gen_instances] wrote {stage2_a8w4_meta_path}")
    print(
        f"[opus_moe gen_instances] wrote {stage1_a8w4_manifest_path} with "
        f"{len(STAGE1_A8W4_KERNELS)} A8W4 stage1 kid(s)"
    )
    print(f"[opus_moe gen_instances] wrote {stage1_a8w4_meta_path}")
    print(
        "[opus_moe gen_instances] wrote 1 runtime-K stage2 device shard, "
        "2 stage1 shard(s), 4 route-reduce shard(s), and 1 BF16 shard"
    )


if __name__ == "__main__":
    main()
