# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from codegen.common import (
    _A16W16_TAGS,
    _GFX942_A16W16_TAGS,
    _NOSPLIT,
    _SPLITK,
    get_arch_map,
    kid_arch as _kid_arch_common,
)

# Import for side-effect: each arch module self-registers into EMIT_REGISTRY
# and ARCH_MAP_REGISTRY at import time.
from codegen import gen_instances_gfx950 as _gfx950  # noqa: F401
from codegen import gen_instances_gfx942 as _gfx942  # noqa: F401
from codegen import gen_instances_gfx1250 as _gfx1250  # noqa: F401
from opus_gemm_common import (
    HEURISTIC_DEFAULT_KIDS,
    OpusGemmInstance,
    heuristic_kids_for_arch,
    a8w8_kernels_list,
    a8w8_scale_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
    a16w16_kernels_list,
    a16w16_mono_tile_kernels_list,
    default_kernels_dict,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    kernels_list,
)

# Cross-arch maps merged from per-arch contributions. Each arch module
# registers its piece into ARCH_MAP_REGISTRY at import; we merge gfx950 first
# (legacy default) then overlay gfx942 entries.
PIPELINE_HEADER_MAP = {
    **get_arch_map("gfx950", "pipeline_header"),
    **get_arch_map("gfx942", "pipeline_header"),
    **get_arch_map("gfx1250", "pipeline_header"),
}

TRAITS_HEADER_MAP = {
    **get_arch_map("gfx950", "traits_header"),
    **get_arch_map("gfx942", "traits_header"),
    **get_arch_map("gfx1250", "traits_header"),
}

KERNEL_FUNC_MAP = {
    **get_arch_map("gfx950", "kernel_func"),
    **get_arch_map("gfx942", "kernel_func"),
    **get_arch_map("gfx1250", "kernel_func"),
}

SPLITK_REDUCE_EXTRA_MAP = {
    "gfx950": get_arch_map("gfx950", "splitk_reduce_extra"),
    "gfx942": get_arch_map("gfx942", "splitk_reduce_extra"),
    "gfx1250": get_arch_map("gfx1250", "splitk_reduce_extra"),
}

SPLITK_REDUCE_ABI_MAP = {
    "gfx950": {
        "forward_decl_include": '#include "gfx950/opus_gemm_traits_a16w16_gfx950.cuh"\n',
        "kernel": "splitk_reduce_kernel",
        "ws_arg": "const opus_splitk_ws_handle* ws_handle",
        "ws_type": "const opus_splitk_ws_handle*",
        "baseline_has_oob": (True, False),
    },
    "gfx942": {
        "forward_decl_include": '#include "gfx942/opus_gemm_traits_a16w16.cuh"\n',
        "kernel": "splitk_reduce_kernel_fallback",
        "ws_arg": "const opus_splitk_ws_handle* ws_handle",
        "ws_type": "const opus_splitk_ws_handle*",
        "baseline_has_oob": (True,),
    },
    "gfx1250": {
        # gfx1250 cluster/TDM split-K: fp32 workspace + separate reduce kernel.
        # Distinct kernel NAME (splitk_reduce_kernel_gfx1250) but the same
        # ws_handle ABI as gfx950, so it never collides in a multi-arch build.
        "forward_decl_include": '#include "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh"\n',
        "kernel": "splitk_reduce_kernel_gfx1250",
        "ws_arg": "const opus_splitk_ws_handle* ws_handle",
        "ws_type": "const opus_splitk_ws_handle*",
        "baseline_has_oob": (True, False),
    },
}

SPLITK_REDUCE_ARCHES = tuple(SPLITK_REDUCE_ABI_MAP)
LEGACY_OPUS_ARCH = "gfx950"


def _splitk_reduce_baseline_instantiations(reduce_kernel, ws_ptr_type, has_oob):
    has_oob_str = "true" if has_oob else "false"
    return (
        f"// HAS_OOB={has_oob_str} variants\n"
        f"template __global__ void {reduce_kernel}<16, 64, __bf16, true,  __bf16, {has_oob_str}>(\n"
        f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
        f"    const __bf16*, int);\n"
        f"template __global__ void {reduce_kernel}<16, 64, __bf16, false, __bf16, {has_oob_str}>(\n"
        f"    {ws_ptr_type}, __bf16*, int, int, int, int, int, int,\n"
        f"    const __bf16*, int);\n"
        f"template __global__ void {reduce_kernel}<16, 64, float,  true,  float,  {has_oob_str}>(\n"
        f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
        f"    const float*,  int);\n"
        f"template __global__ void {reduce_kernel}<16, 64, float,  false, float,  {has_oob_str}>(\n"
        f"    {ws_ptr_type}, float*,  int, int, int, int, int, int,\n"
        f"    const float*,  int);\n"
    )


def _pipeline_header_for(k):
    if getattr(k, "is_4g_safe", False):
        # 4g_safe is gfx950-only (no gfx942 sibling pipeline exists).
        from codegen.gen_instances_gfx950 import PIPELINE_HEADER_MAP_4G_SAFE

        return PIPELINE_HEADER_MAP_4G_SAFE[k.kernel_tag]
    return PIPELINE_HEADER_MAP[k.kernel_tag]


def _kernel_func_for(k):
    if getattr(k, "is_4g_safe", False):
        from codegen.gen_instances_gfx950 import KERNEL_FUNC_MAP_4G_SAFE

        return KERNEL_FUNC_MAP_4G_SAFE[k.kernel_tag]
    return KERNEL_FUNC_MAP[k.kernel_tag]


INPUT_DTYPE_MAP = {
    "a8w8_scale": ("fp8_t", "fp8_t"),
    "a8w8": ("fp8_t", "fp8_t"),
    **{tag: ("bf16_t", "bf16_t") for tag in _A16W16_TAGS},
}

# All a16w16 tags share the 4-arg (XQ, WQ, Y, int splitK) lookup-table slot.
A16W16_TUNE_TAGS = set(_A16W16_TAGS)
# NOSCALE: 3-arg launchers (a16w16 family + a8w8 non-scale).
NOSCALE_TAGS = A16W16_TUNE_TAGS | {"a8w8"}

# SplitK tags live in the <fp32_t> dispatch slot; each instance's traits pick
# the actual workspace dtype and the reduce launcher writes the requested Y.
SPLITK_TAGS = {
    "a16w16_flatmm_splitk",
    "a16w16_cluster_tdm_splitk_ws",
    "a16w16_clusterlaunch_tdm_splitk_ws",
    *_SPLITK,
}

TRAITS_NAME_MAP = {
    **get_arch_map("gfx950", "traits_name"),
    **get_arch_map("gfx942", "traits_name"),
    **get_arch_map("gfx1250", "traits_name"),
}

KARGS_NAME_MAP = {
    **get_arch_map("gfx950", "kargs_name"),
    **get_arch_map("gfx942", "kargs_name"),
    **get_arch_map("gfx1250", "kargs_name"),
}


def _kargs_template_vars(kernel_tag, kargs_name):
    # Paired W3 kernels: fn arg 'Kargs' so deduction keeps host/device mangling.
    if kernel_tag in _NOSPLIT or kernel_tag in _SPLITK:
        return f", {kargs_name}", ", typename Kargs", "Kargs"
    return "", "", kargs_name


# INSTANCE_IMPL building blocks. Host pass needs torch/optional; RTC/device passes skip them.
_INSTANCE_IMPL_PREAMBLE_TEMPLATE = """// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_tensor.h"
#include "aiter_stream.h"{extra_host_includes}
#include <optional>
#endif"""


def instance_impl_preamble(extra_host_includes=""):
    return _INSTANCE_IMPL_PREAMBLE_TEMPLATE.format(
        extra_host_includes=extra_host_includes
    )


# Fused host TU sees only traits header + fwd decl; avoids layout-helper ODR clash.
_INSTANCE_IMPL_HOST_TU_SPLIT_TEMPLATE = """#ifdef OPUS_FUSED_HOST_TU
#include "{traits_header}"
template<typename Traits{fwd_decl_kargs_tpl}>
__global__ void {kernel_func}({fwd_decl_kargs_fnarg} kargs);
#else
#include "{pipeline_header}"
#endif"""


def instance_impl_host_tu_split(
    traits_header,
    pipeline_header,
    fwd_decl_kargs_tpl,
    kernel_func,
    fwd_decl_kargs_fnarg,
):
    return _INSTANCE_IMPL_HOST_TU_SPLIT_TEMPLATE.format(
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        kernel_func=kernel_func,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
    )


# Launcher signature tails after Y.
A16W16_TUNE_HOST_EXTRA = ",\n    std::optional<aiter_tensor_t>,\n    int"
A8W8_SCALE_HOST_EXTRA = (
    ",\n    std::optional<aiter_tensor_t> x_scale,"
    "\n    std::optional<aiter_tensor_t> w_scale"
)


def _make_host_decl(kid_name, dtype, host_extra_params):
    return (
        f"template void\n"
        f"{kid_name}<{dtype}>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &Y{host_extra_params});\n"
    )


def _make_device_decl(
    kid_name, dtype, kernel_func, kargs_name, kargs_explicit_param=""
):
    return (
        f"template __global__ void {kernel_func}<\n"
        f"    {kid_name}_Traits<{dtype}>{kargs_explicit_param}>({kargs_name});\n"
    )


def _record_one_instantiation(
    self_obj, k, kernel_func, kargs_name, host_extra, kargs_explicit_param=""
):
    """Record (host_decl, device_decl) for every (kid, dtype) in k.output_dtypes."""
    for CDtype in k.output_dtypes:
        self_obj._host_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "host_decl": _make_host_decl(k.name, CDtype, host_extra),
            }
        )
        self_obj._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "device_decl": _make_device_decl(
                    k.name, CDtype, kernel_func, kargs_name, kargs_explicit_param
                ),
            }
        )


class opus_gemm_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune
        # Compile-time split: Build layout: * One fused HOST TU (instances/all_instances_host.cu)
        # instantiates every launcher's `template...
        self._host_instantiations = []
        self._device_instantiations = []
        self._kid_records = []
        # Pipeline headers for each kernel_tag (used by the per-kid
        # device TU only).
        self._kid_pipeline_header = {}

    # -- Instance generation --

    def gen_instance(self, k: OpusGemmInstance):
        from codegen.gen_instances_gfx942 import (
            _validate_a16w16_em3en4_gfx942,
            _validate_a16w16_gfx942,
            _validate_a16w16_quad_mfma32_gfx942,
            _validate_a16w16_wave_k_coop_gfx942,
        )
        from codegen.gen_instances_gfx950 import (
            _validate_a16w16,
            _validate_a16w16_flatmm,
            _validate_a16w16_flatmm_splitk,
            _validate_a16w16_mono_tile,
            _validate_a16w16_persistent,
        )

        # gfx950 split-barrier (only "a16w16" tag uses this validator).
        if k.kernel_tag == "a16w16":
            info = _validate_a16w16(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        # gfx942 a16w16 family; specialized tags override only the validator.
        elif k.kernel_tag in _GFX942_A16W16_TAGS:
            if k.kernel_tag == "a16w16_em3en4_lds1_pgr2_sk":
                info = _validate_a16w16_em3en4_gfx942(k)
            elif k.kernel_tag in (
                "a16w16_quad_mfma32_kbuf1",
                "a16w16_quad_mfma32_kbuf1_sk",
            ):
                info = _validate_a16w16_quad_mfma32_gfx942(k)
            elif k.kernel_tag in ("a16w16_wave_k_coop", "a16w16_wave_k_coop_accum"):
                info = _validate_a16w16_wave_k_coop_gfx942(k)
            else:
                info = _validate_a16w16_gfx942(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_persistent":
            info = _validate_a16w16_persistent(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  VGPR~{info['vgpr_est']}  AGPR={info['agprs']}"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_mono_tile":
            info = _validate_a16w16_mono_tile(k)
            print(
                f"  {k.name}: E=({info['E_M']},{info['E_N']},{info['E_K']})"
                f"  LDS={info['lds_bytes'] // 1024}KiB"
                f"  K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_flatmm":
            info = _validate_a16w16_flatmm(k)
            print(
                f"  {k.name}: pfk={info['pfk']} "
                f"slots={info['slots']} "
                f"groups=({info['groups_bm']},{info['groups_bn']},{info['groups_bk']}) "
                f"LDS={info['lds_bytes'] // 1024}KiB K>={info['min_k']}"
            )
        elif k.kernel_tag == "a16w16_flatmm_splitk":
            info = _validate_a16w16_flatmm_splitk(k)
            print(
                f"  {k.name}: pfk={info['pfk']} "
                f"slots={info['slots']} "
                f"comrep=({info['com_rep_m']},{info['com_rep_n']}) "
                f"LDS={info['lds_bytes'] // 1024}KiB K>={info['min_k']} WG={k.WG_PER_CU}"
            )

        pipeline_header = _pipeline_header_for(k)
        traits_header = TRAITS_HEADER_MAP[k.kernel_tag]
        kernel_func = _kernel_func_for(k)
        da, db = INPUT_DTYPE_MAP[k.kernel_tag]
        traits_name = TRAITS_NAME_MAP[k.kernel_tag]
        kargs_name = KARGS_NAME_MAP[k.kernel_tag]

        # Track per-kid pipeline header so the per-kid device.cu can include
        # exactly the right one without re-running the full logic.
        self._kid_pipeline_header[k.name] = pipeline_header

        # Dispatch via registry (codegen/common.py EMIT_REGISTRY). Each arch
        # module under codegen/ self-registers (arch, kernel_tag) -> emit fn.
        # Adding a new arch (e.g. gfx1250) = create codegen/gen_instances_gfx1250.py
        # with register_emit("gfx1250", ...) calls + one import in this file.
        from codegen.common import dispatch_emit

        emit_kwargs = dict(
            pipeline_header=pipeline_header,
            traits_header=traits_header,
            kernel_func=kernel_func,
            da=da,
            db=db,
            traits_name=traits_name,
            kargs_name=kargs_name,
            kargs_template_vars=_kargs_template_vars,
            instance_impl_preamble=instance_impl_preamble,
            instance_impl_host_tu_split=instance_impl_host_tu_split,
            record_one_instantiation=_record_one_instantiation,
            make_host_decl=_make_host_decl,
            make_device_decl=_make_device_decl,
            A16W16_TUNE_HOST_EXTRA=A16W16_TUNE_HOST_EXTRA,
            A8W8_SCALE_HOST_EXTRA=A8W8_SCALE_HOST_EXTRA,
            A16W16_TUNE_TAGS=A16W16_TUNE_TAGS,
            BIAS_HOST_VALIDATE=self.BIAS_HOST_VALIDATE,
        )
        dispatch_emit(self, k, **emit_kwargs)

    # Shared host-side bias validation + kargs population. Consumed by gfx950
    # noscale + gfx950 flatmm_splitk + gfx942 splitk emit modules.
    BIAS_HOST_VALIDATE = """
    const void* ptr_bias_ = nullptr;
    int stride_bias_batch_ = 0;
    if (bias.has_value()) {{
        const auto& bt = bias.value();
        AITER_CHECK(bt.is_contiguous(),
            "bias must be contiguous (got non-contiguous tensor)");
        AITER_CHECK(bt.dtype() == Y.dtype(),
            "bias dtype must match Y dtype (got bias=",
            AiterDtype_to_str(bt.dtype()),
            " Y=", AiterDtype_to_str(Y.dtype()), ")");
        if (bt.dim() == 1) {{
            AITER_CHECK(bt.size(0) == N,
                "bias 1D length must equal N (got bias.size(0)=", bt.size(0),
                " N=", N, ")");
            stride_bias_batch_ = 0;
        }} else if (bt.dim() == 2) {{
            AITER_CHECK(bt.size(0) == batch && bt.size(1) == N,
                "bias 2D shape must equal [batch, N] (got [", bt.size(0), ", ",
                bt.size(1), "] vs batch=", batch, " N=", N, ")");
            stride_bias_batch_ = N;
        }} else {{
            AITER_CHECK(false, "bias must be 1D [N] or 2D [batch, N]; got dim=",
                bt.dim());
        }}
        ptr_bias_ = bt.data_ptr();
    }}
"""

    def gen_lookup_dict(self, kernels_dict):
        """Emit opus_gemm_lookup.h with two (M,N,K)->kernel macros.

        Tuned-CSV driven lookup consumed by opus_gemm.cu's runtime
        `opus_dispatch_a16w16<CDataType>`. Two macros (BF16 / FP32)
        mirror `gen_a16w16_tune_lookup` and exist because splitk kids
        (200..210) are only emitted as `<fp32_t>` (their traits
        static_assert D_C==float, so referencing `splitk<bf16_t>`
        produces a linker error).

        Outdtype-aware bucketing
        ------------------------
        kernels_dict tuple keys carry the outdtype string in slot 3
        ((M, N, K, outdtype_str), produced by get_tune_dict). The BF16
        macro picks up rows whose outdtype is "torch.bfloat16" and the
        FP32 macro picks up rows whose outdtype is "torch.float32";
        same-(M,N,K) rows with different outdtypes therefore land in
        different macros and the two C++ maps can resolve to different
        kernels for the same shape. Legacy CSVs without an outdtype
        column are normalized to bf16 by get_tune_dict, so they only
        populate the BF16 map -- matching pre-outdtype-split behavior.

        Per-kid template argument rule:

          * a16w16 kid 4..9         -> `<CTYPE>` (both bf16/fp32 exist).
          * a16w16_flatmm 100..115  -> `<CTYPE>` (both exist).
          * a16w16_flatmm_splitk    -> always `<fp32_t>`. Splitk rows
            with outdtype=bf16 land in the BF16 map (with forced
            <fp32_t> template arg) and rows with outdtype=fp32 land in
            the FP32 map (also with <fp32_t>). Both work because the
            splitk reduce kernel handles the cast / passthrough at
            launch time based on the actual Y dtype.
        """
        # Sorted flat-array layout (was: {(M,N,K), kernel<CTYPE>} initializer list for std::unordered_map).
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_lookup_dict.
//
// Per-CTYPE sorted flat arrays for (M,N,K)->kernel runtime dispatch.
// Same (M,N,K) can resolve to different kernels in the BF16 vs FP32
// tables because get_tune_dict keys winners on (M, N, K, outdtype_str)
// and gen_lookup_dict buckets the rows into per-CTYPE macros below.
// splitk kids appear in either table with their dispatch template forced
// to <fp32_t>; their traits pick the workspace dtype and the reduce
// launcher writes the requested Y dtype.
//
// Lookup is std::lower_bound on the lex-ordered (M, N, K) key. See
// opus_gemm_arch_gfx950.cuh for the dispatch wrapper.
"""

        ENTRY_MATCH_CTYPE = """\
    {{ {{{M}, {N}, {K}}}, &{kernel_name}<CTYPE> }},  \\
"""
        ENTRY_FORCE_FP32 = """\
    {{ {{{M}, {N}, {K}}}, &{kernel_name}<fp32_t> }}, \\
"""

        # Map ctype short name -> CSV outdtype string emitted by the
        # tuner's result_to_df.
        ctype_to_outdtype = {
            "bf16_t": "torch.bfloat16",
            "fp32_t": "torch.float32",
        }

        def _emit_map(f, macro_name: str, ctype: str):
            # No body line break between `\` and the first entry; macro continuation requires every line
            # that participates in the definition ...
            f.write(f"#define {macro_name}(CTYPE) \\\n")
            target_outdtype = ctype_to_outdtype.get(ctype)
            # Collect all (M, N, K, kernel_name, is_splitk) rows for this
            # CTYPE first, so we can sort lex on (M, N, K) before emitting.
            rows = []
            for mnk, k in kernels_dict.items():
                if self.istune and isinstance(mnk, int):
                    # tune mode shouldn't reach here (gen_lookup_dict is
                    # for the runtime (M,N,K) map). Skip defensively.
                    continue
                if not (isinstance(mnk, tuple) and mnk[0] > 0):
                    continue
                if len(mnk) >= 4:
                    row_outdtype = str(mnk[3])
                    if target_outdtype is not None and row_outdtype != target_outdtype:
                        continue
                is_splitk = k.kernel_tag in SPLITK_TAGS
                if not is_splitk and ctype not in k.output_dtypes:
                    continue
                rows.append((int(mnk[0]), int(mnk[1]), int(mnk[2]), k.name, is_splitk))

            rows.sort(key=lambda r: (r[0], r[1], r[2]))
            n = len(rows)
            for i, (M, N, K, name, is_splitk) in enumerate(rows):
                entry = ENTRY_FORCE_FP32 if is_splitk else ENTRY_MATCH_CTYPE
                line = entry.format(M=M, N=N, K=K, kernel_name=name)
                if i == n - 1:
                    # Last entry: drop the trailing `\` so the macro
                    # ends cleanly. Strip the line's continuation.
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        with open(os.path.join(self.working_path, "opus_gemm_lookup.h"), "w") as f:
            f.write(HEADER)
            _emit_map(f, "GENERATE_OPUS_LOOKUP_TABLE_BF16", "bf16_t")
            _emit_map(f, "GENERATE_OPUS_LOOKUP_TABLE_FP32", "fp32_t")

    def gen_a16w16_tune_lookup(self, kernels_dict):
        """Emit opus_gemm_a16w16_tune_lookup.h with int-ID-to-kernel maps for tuning.

        Three a16w16-family tags share the 4-arg launcher signature
        (XQ, WQ, Y, int splitK):
          * a16w16 (split-barrier)      - output_dtypes=["fp32_t", "bf16_t"]
          * a16w16_flatmm (warp-spec)   - output_dtypes=["bf16_t", "fp32_t"]
          * a16w16_flatmm_splitk        - output_dtypes=["fp32_t"] ONLY
            (main kernel writes fp32 workspace; Y=bf16 via reduce kernel.
            Traits static_assert D_C=float, so no <bf16_t> instantiation
            exists for these kids.)

        The bf16 lookup map therefore must NOT reference splitk kids (their
        <bf16_t> specialization is never instantiated -> linker error). The
        dispatcher in opus_gemm.cu forces kid>=200 to the <fp32_t> branch
        anyway, so having them absent from the bf16 map is correct.

        Emit two macros side by side, gated on each kid's output_dtypes set.
        """
        # Same flat-array design as gen_lookup_dict, keyed on int kid instead of (M,N,K).
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_a16w16_tune_lookup.
//
// Per-CTYPE sorted flat arrays for kid->kernel tune dispatch. Kids whose
// output_dtypes doesn't include CTYPE are omitted from that CTYPE's table
// (splitk kids only live in the fp32 table). See
// opus_gemm_arch_gfx950.cuh for the dispatch wrapper.
"""
        ENTRY = """\
    {{ {kid}, &{kernel_name}<CTYPE> }},  \\
"""

        def _emit_map(f, macro_name, ctype):
            f.write(f"#define {macro_name}(CTYPE) \\\n")
            rows = []
            for kid, k in kernels_dict.items():
                if not (isinstance(kid, int) and k.kernel_tag in A16W16_TUNE_TAGS):
                    continue
                if ctype not in k.output_dtypes:
                    continue
                rows.append((kid, k.name))
            rows.sort(key=lambda r: r[0])
            n = len(rows)
            for i, (kid, name) in enumerate(rows):
                line = ENTRY.format(kid=kid, kernel_name=name)
                if i == n - 1:
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        with open(
            os.path.join(self.working_path, "opus_gemm_a16w16_tune_lookup.h"), "w"
        ) as f:
            f.write(HEADER)
            # Use explicit per-CTYPE macro names; the dispatcher in opus_gemm.cu calls the right one from
            # each opus_a16w16_tune_dispatch<CDat...
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_BF16", "bf16_t")
            _emit_map(f, "GENERATE_A16W16_TUNE_LOOKUP_FP32", "fp32_t")

    def gen_manifest_head(self, kernels_dict):
        # Forward declarations for every launcher symbol the dispatcher references.
        MANIFEST_HEAD = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdlib>
#include <optional>
"""
        MANIFEST_SCALE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale);
"""
        # a8w8 noscale (3 args, no splitK): stays compatible with
        # opus_gemm_lookup.h where a8w8 kids live.
        MANIFEST_NOSCALE_3ARG = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y);
"""
        # a16w16 family (5 args with optional bias + splitK): shared signature for tune lookup.
        MANIFEST_NOSCALE_4ARG = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);
"""
        with open(os.path.join(self.working_path, "opus_gemm_manifest.h"), "w") as f:
            f.write(MANIFEST_HEAD)
            for mnk, k in kernels_dict.items():
                if k.kernel_tag in A16W16_TUNE_TAGS:
                    f.write(MANIFEST_NOSCALE_4ARG.format(kernel_name=k.name))
                elif k.kernel_tag in NOSCALE_TAGS:
                    f.write(MANIFEST_NOSCALE_3ARG.format(kernel_name=k.name))
                else:
                    f.write(MANIFEST_SCALE.format(kernel_name=k.name))

    # -- Per-pass TU emission -- Replaces the old "one .cpp per (kid, dtype)" scheme.

    def _emit_fused_host_tu(self):
        """Emit per-arch HOST translation units (one .cu per arch).

        Splitting by arch lets each TU's reduce-kernel forward decl match
        its arch's launcher emit signature.
        In mixed-arch builds (GPU_ARCHS=gfx942;gfx950) a single host TU
        would force one signature for both arches -> no matching function
        for the other arch's launcher -> link / compile fail.

        Per-arch buckets also keep impl-include sets disjoint: gfx950 TU
        only #includes gfx950 kid impl .cuh, etc. ODR clashes between
        same-named layout helpers in different pipeline headers are
        naturally avoided.
        """

        # Bucket host/device instantiations by arch. We classify by the
        # kid_name prefix `opus_gemm_<arch>_*`; legacy kid names without
        # explicit arch prefix default to gfx950 (matches kid_arch).
        def _kid_name_arch(kid_name):
            for ap in SPLITK_REDUCE_ARCHES:
                if kid_name.startswith(f"opus_gemm_{ap}_"):
                    return ap
            return LEGACY_OPUS_ARCH

        host_by_arch = {}
        for row in self._host_instantiations:
            arch = _kid_name_arch(row["kid_name"])
            host_by_arch.setdefault(arch, []).append(row)

        for arch, rows in host_by_arch.items():
            impl_includes = sorted({row["kid_name"] for row in rows})
            host_body = "".join(row["host_decl"] for row in rows)
            reduce_abi = SPLITK_REDUCE_ABI_MAP[arch]
            extra_reduce = SPLITK_REDUCE_EXTRA_MAP.get(arch, {})
            extra_forward_decls = extra_reduce.get("forward_decls", lambda: "")()
            forward_decls = (
                "// Forward declaration only. Specialisations live in per-arch device TUs.\n"
                f"{reduce_abi['forward_decl_include']}"
                "template<int VEC_, int BLOCK_, typename D_OUT,\n"
                "         bool HAS_BIAS_, typename D_BIAS_,\n"
                "         bool HAS_OOB_>\n"
                f"__global__ void {reduce_abi['kernel']}(\n"
                f"    {reduce_abi['ws_arg']}, D_OUT* c_out,\n"
                "    int split_k, int M, int N, int batch,\n"
                "    int padded_M, int padded_N,\n"
                "    const D_BIAS_* bias, int stride_bias_batch);\n"
                f"{extra_forward_decls}"
            )
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                f"// Auto-generated per-arch host TU ({arch}). See gen_instances.py:_emit_fused_host_tu.\n"
                "#ifndef __HIP_DEVICE_COMPILE__\n"
                "#define OPUS_FUSED_HOST_TU 1\n"
                '#include "aiter_tensor.h"\n'
                '#include "aiter_stream.h"\n'
                "#include <optional>\n"
                + forward_decls
                + "".join(f'#include "impl/{name}.cuh"\n' for name in impl_includes)
                + host_body
                + "#endif // host pass only\n"
            )
            Path(
                os.path.join(self.instances_path, f"all_instances_host_{arch}.cu")
            ).write_text(contents)

    def _emit_device_tus(self):
        """Emit one device-only .device.cu per (kid, dtype).

        Each .cu includes the kid's pipeline header (so the kernel
        template body is visible) and explicitly instantiates the
        kernel template. The companion fused host TU's <<<...>>> calls
        end up referencing host stubs that the linker resolves to the
        instantiations here.

        This TU does not include torch -- it doesn't need to, because
        the host pass only sees `template __global__ void k<...>(...)`
        which doesn't depend on any libtorch type. Skipping the torch
        parse on host pass drops each device TU's compile to ~1.5s
        (down from ~13s when torch was forced in).
        """
        for row in self._device_instantiations:
            name = row["kid_name"]
            dtype = row["dtype"]
            # Include the kid's .cuh -- it transitively pulls in the full pipeline header (because
            # OPUS_FUSED_HOST_TU is NOT defined here) an...
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                "// Auto-generated. Do not edit. See gen_instances.py:_emit_device_tus.\n"
                "//\n"
                "// Device-only translation unit for one (kid, dtype) pair.\n"
                "// Compiled with -D__HIPCC_RTC__ (per-source flag in\n"
                "// optCompilerConfig.json) so the host pass takes the\n"
                "// minimal branch -- no torch, no full HIP runtime.\n"
                f'#include "impl/{name}.cuh"\n' + row["device_decl"]
            )
            Path(
                os.path.join(self.instances_path, f"{name}_C{dtype}.device.cu")
            ).write_text(contents)

    def _emit_splitk_reduce_tu(self):
        """Emit a single splitk_reduce.device.cu carrying the 4 reduce
        kernel specialisations (D_OUT bf16/fp32 x HAS_BIAS true/false).

        Why a dedicated TU: each splitk kid's fused-host launcher body
        does <<<...>>> on all 4 reduce specialisations to handle every
        Y dtype / bias combination at runtime. That used to inline the
        4 `template __global__` instantiations into every splitk kid's
        device.cu (see _gen_flatmm_splitk_instance comment). The linker
        deduped the resulting weak symbols, but each splitk TU still
        paid the full RA + ISA-emit cost on its own compile -- ~0.4s
        wall per TU x 23 splitk TUs = ~9s of duplicated CPU work that
        also lengthened each TU's individual wall and tightened the
        ninja schedule on the slowest splitk kid.

        Centralising them here means:
          * each splitk device.cu only carries its own main-kernel
            instantiation (~50% smaller .o, ~0.3-0.5s less wall each),
          * one new tiny TU compiles the 4 reduces in ~1s wall total,
          * link still works because the reduce symbols are __global__
            (the host stubs the fused TU emits are linked against this
            single TU's GPU code, not against per-splitk-TU copies).

        The reduce kernel template lives in splitk_reduce_{arch}.cuh,
        with one header per arch. gfx950 keeps the legacy
        `splitk_reduce_kernel` name; gfx942 names its baseline path
        `splitk_reduce_kernel_fallback` because exact-N row-block reduce
        is the preferred fast path when its constraints hold.
        """
        # Bucket present archs from splitk kids.
        present_archs = set()
        for row in self._device_instantiations:
            name = row["kid_name"]
            for arch_prefix in SPLITK_REDUCE_ARCHES:
                if f"opus_gemm_{arch_prefix}_splitk_" in name:
                    present_archs.add(arch_prefix)
                    break
            else:
                if "splitk" in name:
                    present_archs.add(LEGACY_OPUS_ARCH)

        # Emit one reduce device TU per arch.
        for reduce_arch in sorted(present_archs):
            reduce_header = f"{reduce_arch}/splitk_reduce_{reduce_arch}.cuh"
            reduce_abi = SPLITK_REDUCE_ABI_MAP[reduce_arch]
            ws_ptr_type = reduce_abi["ws_type"]
            reduce_kernel = reduce_abi["kernel"]
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                f"// Auto-generated per-arch reduce TU ({reduce_arch}). See gen_instances.py:_emit_splitk_reduce_tu.\n"
                f'#include "{reduce_header}"\n'
                + "".join(
                    _splitk_reduce_baseline_instantiations(
                        reduce_kernel, ws_ptr_type, has_oob
                    )
                    for has_oob in reduce_abi["baseline_has_oob"]
                )
            )
            extra_reduce = SPLITK_REDUCE_EXTRA_MAP.get(reduce_arch, {})
            contents += extra_reduce.get("device_instantiations", lambda: "")()
            Path(
                os.path.join(
                    self.instances_path, f"splitk_reduce_{reduce_arch}.device.cu"
                )
            ).write_text(contents)

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        # Reset the instantiation accumulators so reruns under the same
        # codegen object don't double-emit.
        self._host_instantiations = []
        self._device_instantiations = []

        for mnk, k in kernels_dict.items():
            self.gen_instance(k)

        # Emit one fused HOST TU + N device TUs (one per kid, dtype) + one dedicated splitk_reduce.device.cu.
        self._emit_fused_host_tu()
        self._emit_device_tus()
        # Only emit the standalone reduce TU if the build actually has a splitk kid (otherwise the fused
        # host TU will never reference any...
        needs_reduce_tu = any(
            ("flatmm_splitk" in row["kid_name"]) or ("_splitk_" in row["kid_name"])
            for row in self._device_instantiations
        )
        if needs_reduce_tu:
            self._emit_splitk_reduce_tu()

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)
        self.gen_a16w16_tune_lookup(kernels_dict)


def get_tune_dict(tune_dict_csv):
    """Load a tuned CSV into the lookup-dict shape consumed by gen_lookup_dict.

    Key layout
    ----------
    Tuple keys: (M, N, K, outdtype_str). Promoting outdtype into the key
    is what lets a single (M, N, K) shape carry distinct winners for bf16
    vs fp32 output (the underlying main kernel hardware rules differ
    enough that the best kid is not always the same; e.g. fp32 output
    biases reduce-bound shapes toward larger split-K). gen_lookup_dict
    then writes outdtype="torch.bfloat16" rows only into the BF16 (M,N,K)
    map and outdtype="torch.float32" rows only into the FP32 (M,N,K) map.

    Backwards compat
    ----------------
    Legacy CSVs without an `outdtype` column are interpreted as
    bf16-output (matches what the tuner used to write). int keys from
    default_kernels_dict are passed through untouched -- gen_lookup_dict
    skips them via the `isinstance(mnk, tuple) and mnk[0] > 0` guard.
    """
    tune_dict = default_kernels_dict
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        if torch.cuda.is_available():
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            tune_df = tune_df[tune_df["cu_num"] == cu_num].reset_index()
        # Accept either the legacy "kernelId" column or the new "solidx" column (matches
        # aiter/configs/model_configs/gptoss_bf16_tuned_ge...
        kid_col = "solidx" if "solidx" in tune_df.columns else "kernelId"
        has_outdtype = "outdtype" in tune_df.columns
        for i in range(len(tune_df)):
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            outdtype = (
                str(tune_df.loc[i, "outdtype"]) if has_outdtype else "torch.bfloat16"
            )
            kid = int(tune_df.loc[i, kid_col])
            if kid in kernels_list:
                tune_dict[(M, N, K, outdtype)] = kernels_list[kid]
    return tune_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for opus GEMM kernel instances",
    )

    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="generate all kernel instances for tuning (id-based lookup)",
    )

    parser.add_argument(
        "--kernel_tag",
        default=None,
        required=False,
        help="filter kernels by tag (e.g. a16w16, a16w16_flatmm, a16w16_flatmm_splitk, a8w8, a8w8_scale)",
    )

    parser.add_argument(
        "--tune_files",
        default=None,
        required=False,
        help=(
            "Colon-separated list of glob patterns pointing at tuned BF16 "
            "GEMM CSVs (e.g. aiter/configs/bf16_tuned_gemm.csv and "
            "aiter/configs/model_configs/*_bf16_tuned_gemm.csv). Each "
            "file is filtered by `libtype == 'opus'`; surviving rows "
            "contribute their `solidx` to the subset-compile set S and "
            "are also baked into opus_gemm_lookup.h via "
            "GENERATE_OPUS_LOOKUP_TABLE_*. Without this flag we still "
            "generate a working module (only HEURISTIC_DEFAULT_KIDS + "
            "sidecar contents), the lookup table stays empty, and the "
            "C++ dispatch falls through to the heuristic for every "
            "untuned shape."
        ),
    )

    parser.add_argument(
        "--compiled_kids_sidecar",
        default=None,
        required=False,
        help=(
            "Path to the subset-compile sidecar (JSON list of int kids). "
            "Defaults to {working_path}/compiled_kids.json. The sidecar "
            "captures the union of CSV opus rows + previous sidecar "
            "contents + HEURISTIC_DEFAULT_KIDS so subsequent rebuilds "
            "are idempotent (no rebuild if every required kid is already "
            "in the .so). gradlib's GemmTuner and opus_gemm_tune.py "
            "expand this sidecar in tuner-startup to add new kids before "
            "triggering an AITER_REBUILD."
        ),
    )

    # Legacy --tune_file alias kept for backward compat with any existing
    # invocations / scripts. Treated as `--tune_files <path>`.
    parser.add_argument(
        "--tune_file",
        default=None,
        required=False,
        help="[DEPRECATED] alias for --tune_files (single path). Use --tune_files instead.",
    )

    args = parser.parse_args()
    if args.tune_files is None and args.tune_file is not None:
        args.tune_files = args.tune_file
    TAG_TO_LIST = {
        "a8w8_scale": a8w8_scale_kernels_list,
        "a8w8": a8w8_kernels_list,
        "a16w16": a16w16_kernels_list,
        "a16w16_flatmm": a16w16_flatmm_kernels_list,
        "a16w16_flatmm_splitk": a16w16_flatmm_splitk_kernels_list,
        "a16w16_mono_tile": a16w16_mono_tile_kernels_list,
        # gfx942 kid range (10000+); two-bucket registry: nosplit + splitk.
        "gfx942_nosplit": gfx942_nosplit_kernels_list,
        "gfx942_splitk": gfx942_splitk_kernels_list,
    }

    # --- Compute the subset-compile set S ------------------------------------ S = (CSV opus rows'
    # kids) ?

    def _expand_tune_paths(spec):
        out = []
        seen = set()
        if not spec:
            return out
        for pat in str(spec).split(os.pathsep):
            pat = pat.strip()
            if not pat:
                continue
            for path in sorted(glob.glob(pat)):
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
        return out

    csv_kids: set[int] = set()
    csv_paths = _expand_tune_paths(args.tune_files)
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        if "libtype" not in df.columns:
            continue
        df = df[df["libtype"] == "opus"]
        if df.empty:
            continue
        kid_col = (
            "solidx"
            if "solidx" in df.columns
            else ("kernelId" if "kernelId" in df.columns else None)
        )
        if kid_col is None:
            continue
        for v in df[kid_col].dropna().tolist():
            try:
                csv_kids.add(int(v))
            except (TypeError, ValueError):
                continue

    sidecar_path = args.compiled_kids_sidecar or os.path.join(
        args.working_path, "compiled_kids.json"
    )
    sidecar_kids: set[int] = set()
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                sidecar_kids = set(int(x) for x in json.load(f))
        except (OSError, ValueError):
            sidecar_kids = set()

    # The compile set: union, intersected with valid kernels_list entries.
    valid_kids = set(kernels_list.keys())
    S = (csv_kids | sidecar_kids | set(HEURISTIC_DEFAULT_KIDS)) & valid_kids

    # Per-arch filter: drop kids whose arch_prefix is not in the target build set.
    _kid_arch = _kid_arch_common

    target_arches = None
    gpu_archs_env = os.getenv("GPU_ARCHS", "native").strip()
    explicit = [
        a.strip().lower()
        for a in gpu_archs_env.split(";")
        if a.strip() and a.strip().lower() != "native"
    ]
    if explicit:
        target_arches = set(explicit)
    else:
        # GPU_ARCHS=native: probe live GPU; skip filter if rocminfo unavailable.
        try:
            from aiter.jit.utils.chip_info import get_gfx_runtime

            target_arches = {get_gfx_runtime().lower()}
        except Exception:
            target_arches = None

    if target_arches is not None:
        before = len(S)
        S = {kid for kid in S if _kid_arch(kernels_list[kid]) in target_arches}
        dropped = before - len(S)
        print(
            f"[opus gen_instances] arch filter: target={sorted(target_arches)} "
            f"dropped {dropped} off-arch kids from |S|"
        )

    # Emit OPUS_BUILD_HAS_* macros so opus_gemm.cu can gate per-arch dispatch
    # tables: a single-arch build (GPU_ARCHS=gfx950) must not link gfx942
    # launcher symbols and vice versa.
    archs_for_header = (
        sorted(target_arches)
        if target_arches is not None
        else ["gfx942", "gfx950", "gfx1250"]
    )
    with open(os.path.join(args.working_path, "opus_build_archs.h"), "w") as f:
        f.write(
            "// SPDX-License-Identifier: MIT\n"
            "// Auto-generated. See gen_instances.py.\n"
            "#pragma once\n"
        )
        for a in archs_for_header:
            f.write(f"#define OPUS_BUILD_HAS_{a.upper()} 1\n")

    # a8w8 (kid 1, 2) referenced unconditionally by dispatcher; symbols must exist on every arch.
    S |= set(a8w8_scale_kernels_list.keys())
    S |= set(a8w8_kernels_list.keys())

    # Honor --kernel_tag as a developer override that *further restricts* the set (within the a16w16
    # / a8w8 families).
    if args.kernel_tag:
        tag_keys = set(TAG_TO_LIST.get(args.kernel_tag, {}).keys())
        if tag_keys:
            # Restrict to the requested family + heuristic defaults + a8w8 dispatch.
            S = (S & tag_keys) | set(HEURISTIC_DEFAULT_KIDS)
            S |= set(a8w8_scale_kernels_list.keys())
            S |= set(a8w8_kernels_list.keys())

    # Heuristic-fallback invariant (single source of truth: opus_gemm_common.py).
    required_heuristic = set(heuristic_kids_for_arch(target_arches))
    missing_heuristic = required_heuristic - S
    assert not missing_heuristic, (
        f"Subset-compile error: heuristic-fallback kids "
        f"{sorted(missing_heuristic)} are missing from the compile set S; "
        f"opus_a16w16_heuristic_kid_gfx950() would return an unbakeable "
        f"kid. Add them to the compile set or update HEURISTIC_DEFAULT_KIDS "
        f"in csrc/opus_gemm/opus_gemm_common.py."
    )

    # Build the per-kid dict that drives codegen.
    kdict = {kid: kernels_list[kid] for kid in sorted(S)}

    print(
        f"[opus gen_instances] subset compile: |S|={len(S)} kids "
        f"(CSV={len(csv_kids)}, sidecar={len(sidecar_kids)}, heuristic={len(HEURISTIC_DEFAULT_KIDS)})"
    )

    codegen = opus_gemm_codegen(args.working_path, args.tune)
    codegen.gen_instances(kdict)

    # Bake the (M, N, K) -> kernel runtime lookup.
    if csv_paths:
        # Concatenate all opus rows from all matched CSV files (filtered by libtype).
        combined_frames = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                continue
            if "libtype" not in df.columns:
                continue
            df = df[df["libtype"] == "opus"]
            if df.empty:
                continue
            # Drop off-arch kids: lookup must only reference symbols S actually emitted.
            if "solidx" in df.columns:
                df = df[df["solidx"].astype(int).isin(S)]
                if df.empty:
                    continue
            combined_frames.append(df)

        if combined_frames:
            combined = pd.concat(combined_frames, ignore_index=True).drop_duplicates()
            tmp_csv = os.path.join(args.working_path, "_combined_opus_tuned.csv")
            combined.to_csv(tmp_csv, index=False)
            tune_dict = get_tune_dict(tmp_csv)
            try:
                os.remove(tmp_csv)
            except OSError:
                pass
            # Filter tune_dict entries to those whose kid is in S (defense
            # in depth -- valid_kids should have already caught everything).
            filtered = {}
            for k, v in tune_dict.items():
                if isinstance(k, tuple) and k[0] > 0:
                    # Find the kid for this entry by reverse-lookup against S.
                    filtered[k] = v
                else:
                    filtered[k] = v  # default_kernels_dict negative-int entries
            codegen.gen_lookup_dict(filtered)
            n_real = sum(1 for k in filtered if isinstance(k, tuple) and k[0] > 0)
            print(
                f"[opus gen_instances] baked {n_real} tuned entries from "
                f"{len(csv_paths)} CSV file(s) into opus_gemm_lookup.h"
            )
        else:
            print(
                f"[opus gen_instances] no `libtype=='opus'` rows found in "
                f"{len(csv_paths)} CSV file(s); using empty lookup"
            )
    elif args.tune_files:
        print(
            f"[opus gen_instances] --tune_files {args.tune_files} matched no "
            f"existing files; using empty lookup"
        )

    # Persist the expanded compile set so subsequent rebuilds reuse it.
    try:
        os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
    except OSError:
        pass
    with open(sidecar_path, "w") as f:
        json.dump(sorted(S), f)
    print(f"[opus gen_instances] wrote sidecar with {len(S)} kids: {sidecar_path}")
