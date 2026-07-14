# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate Opus MoE stage2 dispatch headers.

This is intentionally smaller than ``csrc/opus_gemm/gen_instances.py`` today:
the stage2 kernels still live in one header, but the generated manifest is the
single source of truth for kid -> launcher mapping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from opus_moe_common import (  # noqa: E402
    OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT,
    OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT,
    OPUS_A8W4_OUT_MODE_ATOMIC,
    OPUS_A8W4_ROUTE_REDUCE_INSTANCES,
    OPUS_A8W4_SHAPE_FAMILY_CONTRACTS,
    STAGE2_A8W4_KERNELS,
    STAGE2_BF16_KERNELS,
    opus_a8w4_decode_kid,
)

MANIFEST_HEADER = """#pragma once
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


# ---- BF16 private manifest -------------------------------------------------


def _emit_bf16_manifest_header() -> str:
    lines = [MANIFEST_HEADER]
    bf16_kernels = [STAGE2_BF16_KERNELS[kid] for kid in sorted(STAGE2_BF16_KERNELS)]

    lines.append(f"#define OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP_SIZE {len(bf16_kernels)}\n")
    if not bf16_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP\n\n")
    else:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP \\\n")
        for idx, inst in enumerate(bf16_kernels):
            suffix = " \\\n" if idx != len(bf16_kernels) - 1 else "\n"
            lines.append(
                "    {"
                f"{inst.kid}, "
                f"&{inst.launcher}<"
                f"{inst.trait}>"
                "}," + suffix
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


def _cpp_contract_alias(name: str) -> str:
    alias_name = str(name)
    if alias_name.startswith("a8w4_"):
        alias_name = alias_name[len("a8w4_") :]
    suffix = _cpp_name_suffix(alias_name)
    return f"OpusMoeStage2A8W4{suffix}Contract"


# ---- A8W4 metadata and dispatch manifests ---------------------------------


def _emit_a8w4_meta_header() -> str:
    lines = [A8W4_META_HEADER]
    k = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    default_family = OPUS_A8W4_DEFAULT_SHAPE_FAMILY_CONTRACT
    shape_families = sorted(
        OPUS_A8W4_SHAPE_FAMILY_CONTRACTS.values(), key=lambda s: s.name
    )
    a8w4_kernels = [STAGE2_A8W4_KERNELS[kid] for kid in sorted(STAGE2_A8W4_KERNELS)]
    unsupported_shape_families = sorted(
        {
            inst.shape_family
            for inst in a8w4_kernels
            if inst.shape_family not in OPUS_A8W4_SHAPE_FAMILY_CONTRACTS
        }
    )
    if unsupported_shape_families:
        raise ValueError(
            "unsupported Opus A8W4 shape family contract(s): "
            + ", ".join(unsupported_shape_families)
        )
    unsupported_kernel_contracts = sorted(
        {
            inst.kernel_contract
            for inst in a8w4_kernels
            if inst.kernel_contract != k.name
        }
    )
    if unsupported_kernel_contracts:
        raise ValueError(
            "unsupported Opus A8W4 kernel contract(s): "
            + ", ".join(unsupported_kernel_contracts)
        )
    block_ms = sorted({inst.block_m for inst in a8w4_kernels})
    block_ns = sorted({inst.block_n for inst in a8w4_kernels})
    mode_default_keys = [
        (inst.shape_family, inst.out_mode, inst.block_m)
        for inst in a8w4_kernels
        if inst.mode_default
    ]
    duplicate_mode_defaults = sorted(
        {key for key in mode_default_keys if mode_default_keys.count(key) > 1}
    )
    if duplicate_mode_defaults:
        raise ValueError(
            "duplicate Opus A8W4 mode default(s): "
            + ", ".join(
                f"shape_family={shape_family},out_mode={out_mode},block_m={block_m}"
                for shape_family, out_mode, block_m in duplicate_mode_defaults
            )
        )
    k_step_packed = k.bk_logical // k.fp4_values_per_byte
    if k.bk_logical % k.fp4_values_per_byte != 0:
        raise ValueError(
            "Opus A8W4 kernel contract requires bk_logical divisible by "
            "fp4_values_per_byte"
        )
    for family in shape_families:
        if family.logical_inter_dim % k.scale_group_logical_k != 0:
            raise ValueError(
                f"shape family {family.name} logical_inter_dim must be divisible "
                f"by scale_group_logical_k={k.scale_group_logical_k}"
            )
        if family.effective_inter_dim <= 0:
            raise ValueError(
                f"shape family {family.name} effective_inter_dim must be positive"
            )
        if family.effective_inter_dim % k_step_packed != 0:
            raise ValueError(
                f"shape family {family.name} effective_inter_dim must be divisible "
                f"by K_STEP_PACKED={k_step_packed}"
            )
    auto_reduce_model_dims = [
        model_dim
        for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES
        for model_dim in inst.auto_model_dims
    ]
    duplicate_auto_reduce_model_dims = sorted(
        {
            model_dim
            for model_dim in auto_reduce_model_dims
            if auto_reduce_model_dims.count(model_dim) > 1
        }
    )
    if duplicate_auto_reduce_model_dims:
        raise ValueError(
            "duplicate Opus A8W4 route-reduce auto model_dim(s): "
            + ", ".join(
                str(model_dim) for model_dim in duplicate_auto_reduce_model_dims
            )
        )

    lines.extend(
        [
            "template<int LogicalInterDim, int InterDimPad>\n",
            "struct OpusMoeStage2A8W4DecodeContract\n{\n",
            "    static constexpr int DECODE_LOGICAL_INTER_DIM = LogicalInterDim;\n",
            "    static constexpr int DECODE_INTER_DIM_PAD = InterDimPad;\n",
            "    static constexpr int DECODE_EFFECTIVE_INTER_DIM = "
            "LogicalInterDim - InterDimPad;\n",
            "};\n\n",
        ]
    )
    for family in shape_families:
        lines.append(
            f"using {_cpp_contract_alias(family.name)} = "
            "OpusMoeStage2A8W4DecodeContract<"
            f"{family.logical_inter_dim}, {family.inter_dim_pad}>;\n"
        )
    lines.extend(
        [
            "using OpusMoeStage2A8W4DefaultContract = "
            f"{_cpp_contract_alias(default_family.name)};\n\n",
            f"constexpr int kStage2MXFP4ScaleGroupLogicalK = {k.scale_group_logical_k};\n",
            "constexpr int kStage2A8W4DecodeLogicalInterDim = "
            "OpusMoeStage2A8W4DefaultContract::DECODE_LOGICAL_INTER_DIM;\n",
            "constexpr int kStage2A8W4DecodeInterDimPad = "
            "OpusMoeStage2A8W4DefaultContract::DECODE_INTER_DIM_PAD;\n",
        ]
    )
    for block_m in block_ms:
        lines.append(f"constexpr int kStage2A8W4DecodeBlockM{block_m} = {block_m};\n")
    for block_n in block_ns:
        lines.append(f"constexpr int kStage2A8W4DecodeBlockN{block_n} = {block_n};\n")
    lines.extend(
        [
            "constexpr int kStage2A8W4DecodeDefaultBlockM = "
            f"kStage2A8W4DecodeBlockM{k.default_block_m};\n",
            "constexpr int kStage2A8W4DecodeDefaultBlockN = "
            f"kStage2A8W4DecodeBlockN{k.default_block_n};\n",
            f"constexpr int kStage2A8W4DecodeDefaultCtaThreads = {k.default_cta_threads};\n",
            f"constexpr int kStage2A8W4DecodeBKLogical = {k.bk_logical};\n",
            f"constexpr int kStage2A8W4DecodeMfmaM = {k.mfma_m};\n",
            f"constexpr int kStage2A8W4DecodeMfmaN = {k.mfma_n};\n",
            f"constexpr int kStage2A8W4DecodeMfmaK = {k.mfma_k};\n",
            f"constexpr int kStage2A8W4DecodeFp4ValuesPerByte = {k.fp4_values_per_byte};\n",
            f"constexpr int kStage2A8W4DecodeVectorBytes = {k.vector_bytes};\n",
            "constexpr int kStage2A8W4DecodeScaleGroupLogicalK = "
            "kStage2MXFP4ScaleGroupLogicalK;\n",
            "constexpr int kStage2A8W4DecodeScaleGroupsPerRowPack = "
            f"{k.scale_groups_per_row_pack};\n",
            "constexpr int kStage2A8W4DecodeScaleWordsPerGroupPack = "
            f"{k.scale_words_per_group_pack};\n",
            f"constexpr int kStage2A8W4DecodeCVec = {k.c_vec};\n",
            f"constexpr int kStage2A8W4DecodeCValuesPerAtomic = {k.c_values_per_atomic};\n\n",
        ]
    )

    for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES:
        suffix = _cpp_name_suffix(inst.name)
        lines.extend(
            [
                f"constexpr int kStage2A8W4RouteReduce{suffix}BlockN = "
                f"{inst.block_n};\n",
                f"constexpr int kStage2A8W4RouteReduce{suffix}Threads = "
                f"{inst.threads};\n",
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
        "constexpr int stage2_a8w4_route_reduce_auto_block_n(int model_dim)\n{\n    switch(model_dim)\n    {\n"
    )
    for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES:
        suffix = _cpp_name_suffix(inst.name)
        for auto_model_dim in inst.auto_model_dims:
            model_dim = (
                f"kStage2A8W4RouteReduce{suffix}BlockN"
                if auto_model_dim == inst.block_n
                else str(auto_model_dim)
            )
            lines.append(
                f"    case {model_dim}: "
                f"return kStage2A8W4RouteReduce{suffix}BlockN;\n"
            )
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr bool stage2_a8w4_block_m_is_valid(int block_m)\n{\n    switch(block_m)\n    {\n"
    )
    for block_m in block_ms:
        lines.append(f"    case {block_m}:\n")
    lines.append("        return true;\n    default: return false;\n    }\n}\n\n")

    lines.append(
        "constexpr bool stage2_a8w4_kid_is_valid(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}:\n")
    lines.append("        return true;\n    default: return false;\n    }\n}\n\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_block_m(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {inst.block_m};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_block_n(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {inst.block_n};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_sort_block_m(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        lines.append(f"    case {inst.kid}: return {inst.sort_block_m};\n")
    lines.append("    default: return -1;\n    }\n}\n\n")

    kid_family_fields = (
        ("logical_inter_dim", "logical_inter_dim"),
        ("inter_dim_pad", "inter_dim_pad"),
        ("effective_inter_dim", "effective_inter_dim"),
    )
    for fn_suffix, attr in kid_family_fields:
        lines.append(
            f"constexpr int stage2_a8w4_kid_{fn_suffix}(int kid)\n"
            "{\n    switch(kid)\n    {\n"
        )
        for inst in a8w4_kernels:
            family = OPUS_A8W4_SHAPE_FAMILY_CONTRACTS[inst.shape_family]
            lines.append(f"    case {inst.kid}: return {getattr(family, attr)};\n")
        lines.append("    default: return -1;\n    }\n}\n\n")

    lines.append(
        "constexpr int stage2_a8w4_kid_k_tiles(int kid)\n{\n    switch(kid)\n    {\n"
    )
    for inst in a8w4_kernels:
        family = OPUS_A8W4_SHAPE_FAMILY_CONTRACTS[inst.shape_family]
        lines.append(
            f"    case {inst.kid}: return {family.effective_inter_dim // k_step_packed};\n"
        )
    lines.append("    default: return -1;\n    }\n}\n\n")

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
        "constexpr int stage2_a8w4_auto_direct_atomic_kid("
        "int logical_inter_dim, int inter_dim_pad, int block_m)\n{\n"
    )
    for family in shape_families:
        lines.append(
            f"    if(logical_inter_dim == {family.logical_inter_dim} && "
            f"inter_dim_pad == {family.inter_dim_pad})\n"
            "    {\n"
            "        switch(block_m)\n"
            "        {\n"
        )
        for block_m in block_ms:
            try:
                kid = opus_a8w4_decode_kid(
                    OPUS_A8W4_OUT_MODE_ATOMIC,
                    block_m,
                    shape_family=family.name,
                )
            except ValueError:
                continue
            lines.append(f"        case {block_m}: return {kid};\n")
        lines.append("        default: return -1;\n        }\n    }\n")
    lines.append("    return -1;\n}\n")
    lines.append(A8W4_META_FOOTER)
    return "".join(lines)


def _emit_a8w4_manifest_header() -> str:
    lines = [A8W4_MANIFEST_HEADER]
    a8w4_kernels = [STAGE2_A8W4_KERNELS[kid] for kid in sorted(STAGE2_A8W4_KERNELS)]

    lines.append(
        f"#define OPUS_MOE_STAGE2_A8W4_DECODE_LOOKUP_SIZE {len(a8w4_kernels)}\n"
    )
    if not a8w4_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES\n")
        return "".join(lines)

    lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES \\\n")
    for idx, inst in enumerate(a8w4_kernels):
        suffix = " \\\n" if idx != len(a8w4_kernels) - 1 else "\n"
        lines.append(
            f"    case {inst.kid}: "
            "return opus_moe_stage2_a8w4_decode_launch_gfx950<"
            "OpusMoeStage2A8W4DecodeShape<"
            f"opus_moe::{_cpp_contract_alias(inst.shape_family)}, "
            f"{inst.block_m}, "
            f"{inst.block_n}, "
            f"{inst.sort_block_m}, "
            f"{_cpp_bool(inst.direct_atomic)}, "
            f"{_cpp_bool(inst.pace_route_blocks_to_pow2)}, "
            f"{inst.block_threads}, "
            f"{inst.min_blocks_per_cu}, "
            f"{inst.cachectl_b}, "
            f"{inst.cachectl_wscale}"
            ">>(kargs, stream);" + suffix
        )
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Opus MoE stage2 dispatch headers"
    )
    parser.add_argument("--working_path", required=True)
    parser.add_argument(
        "--tune_files", default="", help="Accepted for JIT compatibility."
    )
    parser.add_argument(
        "--tune_file", default=None, help="Deprecated alias for --tune_files."
    )
    parser.add_argument(
        "--arch", default=None, help="Optional arch filter, e.g. gfx950"
    )
    parser.add_argument(
        "--cu-num", type=int, default=None, help="Optional CU-count filter"
    )
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_manifest_path = out_dir / "opus_moe_stage2_manifest.h"
    bf16_manifest_path.write_text(_emit_bf16_manifest_header(), encoding="utf-8")
    a8w4_meta_path = out_dir / "opus_moe_stage2_a8w4_meta.h"
    a8w4_meta_path.write_text(_emit_a8w4_meta_header(), encoding="utf-8")
    a8w4_manifest_path = out_dir / "opus_moe_stage2_a8w4_manifest.h"
    a8w4_manifest_path.write_text(_emit_a8w4_manifest_header(), encoding="utf-8")

    print(
        f"[opus_moe gen_instances] wrote {bf16_manifest_path} with "
        f"{len(STAGE2_BF16_KERNELS)} BF16 stage2 kid(s)"
    )
    print(
        f"[opus_moe gen_instances] wrote {a8w4_manifest_path} with "
        f"{len(STAGE2_A8W4_KERNELS)} A8W4 stage2 kid(s)"
    )
    print(f"[opus_moe gen_instances] wrote {a8w4_meta_path}")


if __name__ == "__main__":
    main()
