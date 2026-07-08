# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate Opus MoE stage2 dispatch headers and JIT TUs."""

from __future__ import annotations

import argparse
import shutil
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
    opus_a8w4_decode_kid,
)

A8W4_MANIFEST_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated A8W4 stage2 decode manifest from structured metadata; do not edit.

"""

A8W4_META_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated A8W4 stage2 decode metadata; do not edit.

namespace opus_moe
{

"""

A8W4_META_FOOTER = """
} // namespace opus_moe
"""

A8W4_PIPELINE_HEADER = (
    "gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh"
)
A8W4_TRAITS_HEADER = "gfx950/a8w4/opus_moe_traits_stage2_a8w4_decode_gfx950.cuh"
A8W4_KERNEL_FUNC = "opus_moe_stage2_a8w4_decode_kernel_gfx950"
A8W4_ROUTE_REDUCE_HEADER = (
    "gfx950/opus_moe_stage2_route_output_reduce_kernel_gfx950.cuh"
)
A8W4_ROUTE_REDUCE_KERNEL = (
    "opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950"
)
A8W4_ROUTE_REDUCE_TOPK_SPECIALIZATIONS = (0, 4, 6, 8)
A8W4_ROUTE_REDUCE_SMALL_BLOCK_N = 2048
A8W4_ROUTE_REDUCE_DEFAULT_BLOCK_N = 4096
A8W4_ROUTE_REDUCE_DEFAULT_THREADS = 256
A8W4_ROUTE_REDUCE_DEFAULT_INSTANCES = (
    (A8W4_ROUTE_REDUCE_SMALL_BLOCK_N, A8W4_ROUTE_REDUCE_DEFAULT_THREADS),
    (A8W4_ROUTE_REDUCE_DEFAULT_BLOCK_N, A8W4_ROUTE_REDUCE_DEFAULT_THREADS),
)


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


def _a8w4_launcher_name(kid: int) -> str:
    return f"opus_moe_stage2_a8w4_decode_launch_kid_{int(kid)}_gfx950"


def _a8w4_traits_alias(kid: int) -> str:
    return f"OpusMoeStage2A8W4DecodeKid{int(kid)}Traits"


def _a8w4_impl_filename(kid: int) -> str:
    return f"{_a8w4_launcher_name(kid)}.cuh"


def _a8w4_traits_type(inst) -> str:
    return (
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
        ">"
    )


def _route_reduce_instantiation_rows() -> list[tuple[int, int]]:
    rows = [
        *A8W4_ROUTE_REDUCE_DEFAULT_INSTANCES,
        *(
            (int(inst.block_n), int(inst.threads))
            for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES
        ),
    ]
    return list(dict.fromkeys(rows))


def _a8w4_device_tu_contents(traits_alias: str, traits_type: str) -> str:
    return (
        "// SPDX-License-Identifier: MIT\n"
        "// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.\n"
        "// Auto-generated A8W4 decode device TU; do not edit.\n"
        f'#include "{A8W4_PIPELINE_HEADER}"\n'
        f"using {traits_alias} = {traits_type};\n"
        f"template __global__ void {A8W4_KERNEL_FUNC}<{traits_alias}>(\n"
        "    opus_moe_stage2_a8w4_kargs);\n"
    )


def _append_switch_return(lines, signature, switch_expr, cases, default) -> None:
    lines.append(f"{signature}\n{{\n    switch({switch_expr})\n    {{\n")
    for label, value in cases:
        lines.append(f"    case {label}: return {value};\n")
    lines.append(f"    default: return {default};\n    }}\n}}\n\n")


def _append_switch_true(
    lines: list[str], signature: str, switch_expr: str, labels
) -> None:
    lines.append(f"{signature}\n{{\n    switch({switch_expr})\n    {{\n")
    for label in labels:
        lines.append(f"    case {label}:\n")
    lines.append("        return true;\n    default: return false;\n    }\n}\n\n")


class OpusMoeCodegen:
    def __init__(self, working_path: Path):
        self.working_path = working_path
        self.impl_path = working_path / "impl"
        self.instances_path = working_path / "instances"

    def _prepare_dirs(self) -> None:
        for obsolete in ("opus_moe_stage2_manifest.h",):
            (self.working_path / obsolete).unlink(missing_ok=True)
        for path in (self.impl_path, self.instances_path):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

    def _emit_a8w4_impl(self, inst) -> None:
        kid = int(inst.kid)
        launcher = _a8w4_launcher_name(kid)
        traits_alias = _a8w4_traits_alias(kid)
        traits_type = _a8w4_traits_type(inst)
        impl_name = _a8w4_impl_filename(kid)
        contents = f"""// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated A8W4 stage2 launcher impl; do not edit.
#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_hip_common.h"
#endif

#ifdef OPUS_FUSED_HOST_TU
#include "{A8W4_TRAITS_HEADER}"
template<typename Traits>
__global__ void {A8W4_KERNEL_FUNC}(opus_moe_stage2_a8w4_kargs kargs);
#else
#include "{A8W4_PIPELINE_HEADER}"
#endif

using {traits_alias} = {traits_type};

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
void {launcher}(const opus_moe_stage2_a8w4_kargs& kargs, hipStream_t stream)
{{
    using Traits = {traits_alias};
    int route_blocks =
        (kargs.sorted_blocks * Traits::SORT_BLOCK_M + Traits::B_M - 1) /
        Traits::B_M;
    opus_moe_stage2_a8w4_kargs launch_kargs = kargs;
    launch_kargs.sorted_blocks = route_blocks;
    if constexpr(Traits::DECODE_PACE_ROUTE_BLOCKS_TO_POW2)
    {{
        int paced_route_blocks = 1;
        while(paced_route_blocks < route_blocks)
            paced_route_blocks <<= 1;
        route_blocks = paced_route_blocks;
        launch_kargs.sorted_blocks = route_blocks;
    }}
    AITER_CHECK(kargs.model_dim % Traits::B_N == 0,
                "Opus A8W4 stage2 requires model_dim to be a multiple of block_n=",
                Traits::B_N,
                ", got ",
                kargs.model_dim);
    const int n_tiles = kargs.model_dim / Traits::B_N;
    dim3 grid(n_tiles, route_blocks, 1);
    dim3 block(Traits::BLOCK_SIZE);
    {A8W4_KERNEL_FUNC}<Traits><<<grid, block, 0, stream>>>(launch_kargs);
}}
#endif
"""
        (self.impl_path / impl_name).write_text(contents, encoding="utf-8")

    def _emit_fused_host_tu(self) -> None:
        a8w4_includes = "".join(
            f'#include "impl/{_a8w4_impl_filename(kid)}"\n'
            for kid in sorted(STAGE2_A8W4_KERNELS)
        )
        contents = (
            "// SPDX-License-Identifier: MIT\n"
            "// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.\n"
            "// Auto-generated per-arch host TU (gfx950); do not edit.\n"
            "#ifndef __HIP_DEVICE_COMPILE__\n"
            "#define OPUS_FUSED_HOST_TU 1\n"
            '#include "gfx950/opus_moe_arch_gfx950.cuh"\n'
            + a8w4_includes
            + "#endif // host pass only\n"
        )
        (self.instances_path / "all_instances_host_gfx950.cu").write_text(
            contents, encoding="utf-8"
        )

    def _emit_device_tus(self) -> None:
        for kid in sorted(STAGE2_A8W4_KERNELS):
            (self.instances_path / f"{_a8w4_launcher_name(kid)}.device.cu").write_text(
                _a8w4_device_tu_contents(
                    _a8w4_traits_alias(kid),
                    _a8w4_traits_type(STAGE2_A8W4_KERNELS[kid]),
                ),
                encoding="utf-8",
            )

    def _emit_route_reduce_tu(self) -> None:
        rows = _route_reduce_instantiation_rows()
        lines = [
            "// SPDX-License-Identifier: MIT\n",
            "// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.\n",
            "// Auto-generated route-reduce device TU (gfx950); do not edit.\n",
            f'#include "{A8W4_ROUTE_REDUCE_HEADER}"\n',
        ]
        for block_n, threads in rows:
            for topk in A8W4_ROUTE_REDUCE_TOPK_SPECIALIZATIONS:
                for route_fp8 in (False, True):
                    route_fp8_str = _cpp_bool(route_fp8)
                    lines.append(
                        f"template __global__ void {A8W4_ROUTE_REDUCE_KERNEL}<"
                        f"{block_n}, {threads}, {topk}, {route_fp8_str}>(\n"
                        "    opus_moe_stage2_route_reduce_kargs);\n"
                    )
        (
            self.instances_path / "opus_moe_stage2_route_reduce_gfx950.device.cu"
        ).write_text("".join(lines), encoding="utf-8")

    def gen_instances(self) -> None:
        self._prepare_dirs()
        for kid in sorted(STAGE2_A8W4_KERNELS):
            self._emit_a8w4_impl(STAGE2_A8W4_KERNELS[kid])
        self._emit_fused_host_tu()
        self._emit_device_tus()
        self._emit_route_reduce_tu()


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
            f"constexpr int kStage2A8W4DecodeCValuesPerAtomic = {k.c_values_per_atomic};\n",
            "constexpr int kStage2RouteOutputReduceSmallBlockN = "
            f"{A8W4_ROUTE_REDUCE_SMALL_BLOCK_N};\n",
            "constexpr int kStage2RouteOutputReduceDefaultBlockN = "
            f"{A8W4_ROUTE_REDUCE_DEFAULT_BLOCK_N};\n",
            "constexpr int kStage2RouteOutputReduceDefaultThreads = "
            f"{A8W4_ROUTE_REDUCE_DEFAULT_THREADS};\n\n",
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

    _append_switch_true(
        lines,
        "constexpr bool stage2_a8w4_block_m_is_valid(int block_m)",
        "block_m",
        block_ms,
    )
    _append_switch_true(
        lines,
        "constexpr bool stage2_a8w4_kid_is_valid(int kid)",
        "kid",
        [inst.kid for inst in a8w4_kernels],
    )

    def shape_family(inst):
        return OPUS_A8W4_SHAPE_FAMILY_CONTRACTS[inst.shape_family]

    kid_switches = [
        (
            "constexpr int stage2_a8w4_kid_block_m(int kid)",
            "-1",
            lambda inst: inst.block_m,
        ),
        (
            "constexpr int stage2_a8w4_kid_block_n(int kid)",
            "-1",
            lambda inst: inst.block_n,
        ),
    ]
    kid_family_fields = (
        ("logical_inter_dim", "logical_inter_dim"),
        ("inter_dim_pad", "inter_dim_pad"),
        ("effective_inter_dim", "effective_inter_dim"),
    )
    for fn_suffix, attr in kid_family_fields:
        kid_switches.append(
            (
                f"constexpr int stage2_a8w4_kid_{fn_suffix}(int kid)",
                "-1",
                lambda inst, attr=attr: getattr(shape_family(inst), attr),
            )
        )
    kid_switches.extend(
        [
            (
                "constexpr int stage2_a8w4_kid_k_tiles(int kid)",
                "-1",
                lambda inst: shape_family(inst).effective_inter_dim // k_step_packed,
            ),
            (
                "constexpr bool stage2_a8w4_kid_uses_route_out(int kid)",
                "false",
                lambda inst: _cpp_bool(inst.route_out),
            ),
            (
                "constexpr bool stage2_a8w4_kid_route_fp8(int kid)",
                "false",
                lambda inst: _cpp_bool(inst.route_out_fp8),
            ),
            (
                "constexpr const char* stage2_a8w4_kid_name(int kid)",
                '"unknown"',
                lambda inst: f'"{_cpp_string(inst.name)}"',
            ),
        ]
    )
    for signature, default, value_fn in kid_switches:
        _append_switch_return(
            lines,
            signature,
            "kid",
            [(inst.kid, value_fn(inst)) for inst in a8w4_kernels],
            default,
        )

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

    for inst in a8w4_kernels:
        lines.append(
            f"void {_a8w4_launcher_name(inst.kid)}(\n"
            "    const opus_moe_stage2_a8w4_kargs& kargs,\n"
            "    hipStream_t stream);\n"
        )
    lines.append("\n")

    lines.append(
        "template<int TOPK>\n"
        "inline bool opus_moe_stage2_a8w4_route_reduce_dispatch_generated_gfx950(\n"
        "    const opus_moe_stage2_route_reduce_kargs& kargs,\n"
        "    dim3 grid,\n"
        "    hipStream_t stream,\n"
        "    int block_n)\n"
        "{\n"
        "    switch(block_n)\n"
        "    {\n"
    )
    for inst in OPUS_A8W4_ROUTE_REDUCE_INSTANCES:
        suffix = _cpp_name_suffix(inst.name)
        lines.extend(
            [
                f"    case opus_moe::kStage2A8W4RouteReduce{suffix}BlockN:\n",
                "        opus_moe_stage2_reduce_token_slot_route_output_launch_variant_gfx950<\n",
                f"            opus_moe::kStage2A8W4RouteReduce{suffix}BlockN,\n",
                f"            opus_moe::kStage2A8W4RouteReduce{suffix}Threads,\n",
                "            TOPK>(kargs, grid, stream);\n",
                "        return true;\n",
            ]
        )
    lines.extend(
        [
            "    default:\n",
            "        return false;\n",
            "    }\n",
            "}\n\n",
        ]
    )

    if not a8w4_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES\n")
        return "".join(lines)

    lines.append("#define GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES \\\n")
    for idx, inst in enumerate(a8w4_kernels):
        suffix = " \\\n" if idx != len(a8w4_kernels) - 1 else "\n"
        lines.append(
            f"    case {inst.kid}: "
            f"return {_a8w4_launcher_name(inst.kid)}(kargs, stream);" + suffix
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
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    a8w4_meta_path = out_dir / "opus_moe_stage2_a8w4_meta.h"
    a8w4_meta_path.write_text(_emit_a8w4_meta_header(), encoding="utf-8")
    a8w4_manifest_path = out_dir / "opus_moe_stage2_a8w4_manifest.h"
    a8w4_manifest_path.write_text(_emit_a8w4_manifest_header(), encoding="utf-8")
    codegen = OpusMoeCodegen(out_dir)
    codegen.gen_instances()

    print(
        f"[opus_moe gen_instances] wrote {a8w4_manifest_path} with "
        f"{len(STAGE2_A8W4_KERNELS)} A8W4 stage2 kid(s)"
    )
    print(f"[opus_moe gen_instances] wrote {a8w4_meta_path}")
    print(
        "[opus_moe gen_instances] wrote "
        f"{len(STAGE2_A8W4_KERNELS)} A8W4 impl/device TU(s) and "
        "1 route-reduce device TU"
    )


if __name__ == "__main__":
    main()
