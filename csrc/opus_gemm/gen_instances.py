# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from codegen import gen_instances_gfx942 as _gfx942  # noqa: F401

# Architecture modules register their code emitters at import time.
from codegen import gen_instances_gfx950 as _gfx950  # noqa: F401
from codegen import gen_instances_gfx1250 as _gfx1250  # noqa: F401
from codegen.common import (
    _A16W16_TAGS,
    _GFX942_A16W16_TAGS,
    _NOSPLIT,
    _SPLITK,
    get_arch_map,
)
from codegen.common import (
    kid_arch as _kid_arch_common,
)
from opus_gemm_common import (
    BMM_MXSCALE_KIDS,
    DEFAULT_COMPILED_KIDS,
    OPUS_MANDATORY_A8_KIDS,
    OpusGemmInstance,
    a8w8_kernels_list,
    a8w8_mxscale_bmm_kernel_lists,
    a8w8_scale_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
    a16w16_kernels_list,
    a16w16_mono_tile_kernels_list,
    default_compiled_kids_for_arch,
    gfx942_a8w8_kernels_list,
    gfx942_nosplit_kernels_list,
    gfx942_splitk_kernels_list,
    gfx1250_clusterlaunch_kernels_list,
    gfx1250_4wave_co_kernels_list,
    gfx1250_kernels_list,
    gfx1250_splitk_fuse_kernels_list,
    kernels_list,
)

# Merge the codegen maps registered by each architecture.
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
        "ws_arg": "const void* ws_ptr",
        "ws_type": "const void*",
        "baseline_has_oob": (True, False),
    },
    "gfx942": {
        "forward_decl_include": '#include "gfx942/a16w16/opus_gemm_traits_a16w16.cuh"\n',
        "kernel": "splitk_reduce_kernel_fallback",
        "ws_arg": "const void* ws_ptr",
        "ws_type": "const void*",
        "baseline_has_oob": (True,),
    },
    "gfx1250": {
        # gfx1250 cluster/TDM split-K: exact-kid bf16/fp32 workspace + separate
        # compile-time-split reduce kernel. The shared generator emits both
        # workspace types; gen_instances_gfx1250.py adds mixed fp32-bias/bf16-Y.
        # Distinct kernel NAME (splitk_reduce_kernel_gfx1250) keeps it from
        # colliding with gfx950 in a multi-arch build.
        "forward_decl_include": '#include "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh"\n',
        "kernel": "splitk_reduce_kernel_gfx1250",
        "ws_arg": "const void* ws_ptr",
        "ws_type": "const void*",
        "baseline_has_oob": (True, False),
        "forward_decl_extra_template_params": ", int SPLIT_K_, typename D_WS_",
    },
}

SPLITK_REDUCE_ARCHES = tuple(SPLITK_REDUCE_ABI_MAP)
LEGACY_OPUS_ARCH = "gfx950"


def _kid_name_arch(kid_name):
    """Resolve a generated symbol's owning architecture."""
    for arch_prefix in SPLITK_REDUCE_ARCHES:
        if kid_name.startswith(f"opus_gemm_{arch_prefix}_"):
            return arch_prefix
    return LEGACY_OPUS_ARCH


def _own_arch_device_pass_guard(arch):
    """Admit the host pass and only this kid's owning device pass."""
    return (
        f"#if !defined(__HIP_DEVICE_COMPILE__) || defined(__{arch}__)\n",
        f"#endif // host pass or {arch} device pass\n",
    )


def _splitk_reduce_baseline_instantiations(
    reduce_kernel,
    ws_ptr_type,
    has_oob,
    vec=16,
    block=64,
    split_ks=(None,),
    workspace_types=(None,),
):
    has_oob_str = "true" if has_oob else "false"
    configs = (
        ("__bf16", "true", "__bf16"),
        ("__bf16", "false", "__bf16"),
        ("float", "true", "float"),
        ("float", "false", "float"),
    )
    out = f"// HAS_OOB={has_oob_str} variants\n"
    for split_k in split_ks:
        for workspace_type in workspace_types:
            tail = "" if split_k is None else f", {split_k}, {workspace_type}"
            for out_type, has_bias, bias_type in configs:
                out += (
                    f"template __global__ void {reduce_kernel}<"
                    f"{vec}, {block}, {out_type}, {has_bias}, {bias_type}, "
                    f"{has_oob_str}{tail}>(\n"
                    f"    {ws_ptr_type}, {out_type}*, int, int, int, int, int, int,\n"
                    f"    const {bias_type}*, int);\n"
                )
    return out


# Arches that own an opus_gemm_arch_*.cuh dispatch header, i.e. one set of
# lookup tables each. Every generated lookup macro is emitted once per arch and
# expanded by that arch's header only: the arches disagree on the a16w16
# launcher signature (gfx1250 takes an extra workspace tensor), so one shared
# macro cannot type-check in a mixed-arch build -- gfx950's table would hold
# gfx1250 function pointers and vice versa. Filtering |S| by GPU_ARCHS hid this
# for single-arch builds only.
# A per-arch macro is legitimately empty (an arch whose kids all missed |S|, or
# which has no tuned row for the host's cu_num), which expands to a zero-length
# table -- accepted as a clang extension, and already the case before the split
# for e.g. the gfx1250 fp32 (M,N,K) table in a gfx1250-only build.
LOOKUP_MACRO_ARCHES = ("gfx950", "gfx942", "gfx1250")


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
    "a8w8_mxscale": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_flatmm_splitk": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_fused": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_minterleave": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_mouter": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_mouter_tunable": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_pipeline": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_wave8n2": ("fp8_t", "fp8_t"),
    "a8w8_mxscale_bmm_wave4m2_selfload": ("fp8_t", "fp8_t"),
    "a8w8": ("fp8_t", "fp8_t"),
    "a8w8_blockscale_bpreshuffle_singlebuf": ("fp8_t", "fp8_t"),
    **{tag: ("bf16_t", "bf16_t") for tag in _A16W16_TAGS},
}

# A16W16 uses separate direct-output and workspace launcher tables.
A16W16_KID_DISPATCH_TAGS = set(_A16W16_TAGS)
A8W8_BPRESHUFFLE_TAGS = {"a8w8_blockscale_bpreshuffle_singlebuf"}
# Three-tensor launchers: A16W16 and A8W8 no-scale.
NOSCALE_TAGS = A16W16_KID_DISPATCH_TAGS | {"a8w8"}

# Split-K tags live in the workspace dispatch table and use their existing
# <fp32_t> host specialization; each instance's traits pick the actual
# workspace dtype. Fused kids write Y in-kernel; the other tags launch a
# standalone reducer.
SPLITK_TAGS = {
    "a16w16_flatmm_splitk",
    "a16w16_cluster_tdm_splitk_ws",
    "a16w16_clusterlaunch_tdm_splitk_ws",
    "a16w16_clusterlaunch_tdm_splitk_fuse",
    "a16w16_em3en4_lds1_pgr2_sk",
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
    if kernel_tag in (
        "a8w8_mxscale_bmm_flatmm_splitk",
        "a8w8_mxscale_bmm_fused",
    ):
        return (
            "",
            ", typename D_OUT, bool DIRECT_ONLY, bool PREFETCH_SCALE, bool PRELOAD_SF_LDS",
            kargs_name,
        )
    if kernel_tag == "a8w8_mxscale_bmm_minterleave":
        return "", ", typename D_OUT, bool SKIP_SCALE_WAIT", kargs_name
    if kernel_tag == "a8w8_mxscale_bmm_pipeline":
        return "", "", kargs_name
    if kernel_tag in (
        "a8w8_mxscale_bmm_mouter",
        "a8w8_mxscale_bmm_mouter_tunable",
    ):
        return "", ", typename D_OUT, bool SKIP_SCALE_WAIT", kargs_name
    if kernel_tag == "a8w8_mxscale_bmm_wave8n2":
        return "", ", typename D_OUT", kargs_name
    if kernel_tag == "a8w8_mxscale_bmm_wave4m2_selfload":
        return (
            "",
            ", typename D_OUT, bool SKIP_SCALE_WAIT, bool PACK_SCALE_ON_DEMAND",
            kargs_name,
        )
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


# Extra parameters appended to each generated launcher signature.
A16W16_LAUNCH_HOST_EXTRA = ",\n    std::optional<aiter_tensor_t>,\n    int"
A16W16_WORKSPACE_LAUNCH_HOST_EXTRA = (
    ",\n    aiter_tensor_t &workspace,"
    "\n    std::optional<aiter_tensor_t>,"
    "\n    int"
)
A8W8_BLOCKSCALE_HOST_EXTRA = (
    ",\n    aiter_tensor_t &x_scale," "\n    aiter_tensor_t &w_scale"
)


def _make_host_decl(kid_name, dtype, host_extra_params):
    return (
        f"template void\n"
        f"{kid_name}<{dtype}>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &Y{host_extra_params});\n"
    )


def _make_a8w8_bpreshuffle_host_decl(kid_name, dtype, _host_extra_params):
    """Emit the ``XQ,WQ,x_scale,w_scale,Y`` host declaration."""
    return (
        f"template void\n"
        f"{kid_name}<{dtype}>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &x_scale,\n"
        f"    aiter_tensor_t &w_scale,\n"
        f"    aiter_tensor_t &Y);\n"
    )


def _make_device_decl(
    kid_name, dtype, kernel_func, kargs_name, kargs_explicit_param=""
):
    return (
        f"template __global__ void {kernel_func}<\n"
        f"    {kid_name}_Traits<{dtype}>{kargs_explicit_param}>({kargs_name});\n"
    )


def _record_one_instantiation(
    self_obj,
    k,
    kernel_func,
    kargs_name,
    host_extra,
    kargs_explicit_param="",
    host_decl_factory=_make_host_decl,
):
    """Record (host_decl, device_decl) for every (kid, dtype) in k.output_dtypes."""
    for CDtype in k.output_dtypes:
        self_obj._host_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "host_decl": host_decl_factory(k.name, CDtype, host_extra),
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
        # Compile-time split: Build layout: * One fused HOST TU per arch
        # (instances/all_instances_host_<arch>.cu)
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

        emit_kwargs = {
            "pipeline_header": pipeline_header,
            "traits_header": traits_header,
            "kernel_func": kernel_func,
            "da": da,
            "db": db,
            "traits_name": traits_name,
            "kargs_name": kargs_name,
            "kargs_template_vars": _kargs_template_vars,
            "instance_impl_preamble": instance_impl_preamble,
            "instance_impl_host_tu_split": instance_impl_host_tu_split,
            "record_one_instantiation": _record_one_instantiation,
            "make_host_decl": _make_host_decl,
            "make_device_decl": _make_device_decl,
            "A16W16_LAUNCH_HOST_EXTRA": A16W16_LAUNCH_HOST_EXTRA,
            "A16W16_WORKSPACE_LAUNCH_HOST_EXTRA": (A16W16_WORKSPACE_LAUNCH_HOST_EXTRA),
            "A8W8_BLOCKSCALE_HOST_EXTRA": A8W8_BLOCKSCALE_HOST_EXTRA,
            "make_a8w8_bpreshuffle_host_decl": (_make_a8w8_bpreshuffle_host_decl),
            "A16W16_KID_DISPATCH_TAGS": A16W16_KID_DISPATCH_TAGS,
            "BIAS_HOST_VALIDATE": self.BIAS_HOST_VALIDATE,
        }
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

    def gen_a16w16_kid_dispatch(self, kernels_dict):
        """Emit per-arch A16W16 direct and workspace launcher tables."""
        HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_a16w16_kid_dispatch.
//
// Per-arch sorted flat arrays for strict kid dispatch.  Non-workspace tables
// contain five-argument OpusA16W16Kernel pointers.  Workspace tables contain
// six-argument OpusA16W16WorkspaceKernel pointers. Never combine them with
// the five-argument table.
"""
        NON_WORKSPACE_ENTRY = """\
    {{ {kid}, &{kernel_name}<CTYPE> }},  \\
"""
        WORKSPACE_ENTRY = """\
    {{ {kid}, &{kernel_name}<fp32_t> }},  \\
"""

        def _write_rows(f, macro_name, rows, entry, function_like=False):
            f.write(f"#define {macro_name}_SIZE {len(rows)}\n")
            macro_suffix = "(CTYPE)" if function_like else ""
            if not rows:
                f.write(f"#define {macro_name}{macro_suffix}\n\n")
                return
            f.write(f"#define {macro_name}{macro_suffix} \\\n")
            for index, (kid, name) in enumerate(rows):
                line = entry.format(kid=kid, kernel_name=name)
                if index == len(rows) - 1:
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        def _emit_non_workspace_map(f, arch, ctype):
            rows = []
            for kid, k in kernels_dict.items():
                if not (
                    isinstance(kid, int) and k.kernel_tag in A16W16_KID_DISPATCH_TAGS
                ):
                    continue
                if _kid_arch_common(k) != arch or k.kernel_tag in SPLITK_TAGS:
                    continue
                if ctype not in k.output_dtypes:
                    continue
                if _kid_arch_common(k) != arch:
                    continue
                rows.append((kid, k.name))
            rows.sort(key=lambda r: r[0])
            dtype_suffix = "BF16" if ctype == "bf16_t" else "FP32"
            macro_name = (
                "GENERATE_A16W16_NONWORKSPACE_KID_DISPATCH_"
                f"{arch.upper()}_{dtype_suffix}"
            )
            _write_rows(f, macro_name, rows, NON_WORKSPACE_ENTRY, function_like=True)

        def _emit_workspace_map(f, arch):
            rows = []
            for kid, k in kernels_dict.items():
                if not (
                    isinstance(kid, int) and k.kernel_tag in A16W16_KID_DISPATCH_TAGS
                ):
                    continue
                if _kid_arch_common(k) != arch or k.kernel_tag not in SPLITK_TAGS:
                    continue
                if "fp32_t" not in k.output_dtypes:
                    raise ValueError(
                        f"workspace kid {kid} ({k.name}) has no fp32_t host "
                        "specialization"
                    )
                rows.append((kid, k.name))
            rows.sort(key=lambda r: r[0])
            macro_name = f"GENERATE_A16W16_WORKSPACE_KID_DISPATCH_{arch.upper()}"
            _write_rows(f, macro_name, rows, WORKSPACE_ENTRY)

        with open(
            os.path.join(self.working_path, "opus_gemm_a16w16_kid_dispatch.h"), "w"
        ) as f:
            f.write(HEADER)
            for arch in SPLITK_REDUCE_ARCHES:
                _emit_non_workspace_map(f, arch, "bf16_t")
                _emit_non_workspace_map(f, arch, "fp32_t")
                _emit_workspace_map(f, arch)

    def gen_a8w8_kid_dispatch(self, kernels_dict):
        """Emit sorted A8W8 launcher tables for each interface."""
        header = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See gen_instances.py:gen_a8w8_kid_dispatch.
//
// Interfaces remain separate even when argument counts match. Missing kernels
// use a size-0 table.
"""
        entry = """\
    {{ {kid}, &{kernel_name}<{ctype}> }},  \\
"""

        def _rows(arch, tags, ctype):
            rows = []
            for kid, k in kernels_dict.items():
                if not isinstance(kid, int) or k.kernel_tag not in tags:
                    continue
                if _kid_arch_common(k) != arch or ctype not in k.output_dtypes:
                    continue
                rows.append((kid, k.name))
            rows.sort(key=lambda row: row[0])
            return rows

        def _emit_map(f, macro_name, rows, ctype):
            f.write(f"#define {macro_name}_SIZE {len(rows)}\n")
            if not rows:
                f.write(f"#define {macro_name}\n\n")
                return
            f.write(f"#define {macro_name} \\\n")
            for index, (kid, name) in enumerate(rows):
                line = entry.format(kid=kid, kernel_name=name, ctype=ctype)
                if index == len(rows) - 1:
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

        with open(
            os.path.join(self.working_path, "opus_gemm_a8w8_kid_dispatch.h"), "w"
        ) as f:
            f.write(header)
            _emit_map(
                f,
                "GENERATE_A8W8_NOSCALE_KID_DISPATCH_GFX950",
                _rows("gfx950", {"a8w8"}, "fp32_t"),
                "fp32_t",
            )
            _emit_map(
                f,
                "GENERATE_A8W8_BLOCKSCALE_KID_DISPATCH_GFX950",
                _rows("gfx950", {"a8w8_scale"}, "fp32_t"),
                "fp32_t",
            )
            for arch in SPLITK_REDUCE_ARCHES:
                for ctype, dtype_suffix in (
                    ("bf16_t", "BF16"),
                    ("fp32_t", "FP32"),
                ):
                    macro_name = (
                        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_"
                        f"{arch.upper()}_{dtype_suffix}"
                    )
                    _emit_map(
                        f,
                        macro_name,
                        _rows(arch, A8W8_BPRESHUFFLE_TAGS, ctype),
                        ctype,
                    )

    def gen_bmm_mxscale_kid_dispatch(self):
        """Emit the global exact-kid table for gfx950 MXFP8 BMM launchers."""
        header = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// Auto-generated. Do not edit.
"""
        entry = """\
    {{ {kid}, &{kernel_name}<CTYPE> }},  \\
"""

        rows = sorted(
            (kid, instance.name)
            for family in a8w8_mxscale_bmm_kernel_lists
            for kid, instance in family.items()
            if "fp32_t" in instance.output_dtypes
        )
        with open(
            os.path.join(self.working_path, "opus_bmm_mxscale_kid_dispatch.h"),
            "w",
        ) as f:
            f.write(header)
            f.write(f"#define GENERATE_BMM_MXSCALE_KID_DISPATCH_SIZE {len(rows)}\n")
            f.write("#define GENERATE_BMM_MXSCALE_KID_DISPATCH(CTYPE) \\\n")
            for index, (kid, name) in enumerate(rows):
                line = entry.format(kid=kid, kernel_name=name)
                if index == len(rows) - 1:
                    line = line.rstrip().rstrip("\\").rstrip() + "\n"
                f.write(line)
            f.write("\n")

    def gen_manifest_head(self, kernels_dict):
        # Forward declarations for every launcher symbol the dispatcher references.
        MANIFEST_HEAD = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdlib>
#include <optional>
"""
        MANIFEST_BLOCKSCALE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale);
"""
        MANIFEST_BLOCKSCALE_BPRESHUFFLE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    aiter_tensor_t &Y);
"""
        # a8w8 noscale (3 args, no splitK) has its own exact-kid table.
        MANIFEST_NOSCALE_3ARG = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y);
"""
        # Non-workspace a16w16 launchers keep the existing five-argument ABI.
        MANIFEST_A16W16 = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int splitK);
"""
        # External-workspace launchers receive a caller-owned typed tensor.
        # This covers both two-stage reducers and gfx1250 fused in-cluster
        # reduction.
        MANIFEST_A16W16_WORKSPACE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &workspace,
    std::optional<aiter_tensor_t> bias,
    int splitK);
"""
        MANIFEST_BMM_MXSCALE = """
template <typename D_C>
void
{kernel_name}(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    aiter_tensor_t &x_scale,
    aiter_tensor_t &w_scale,
    std::optional<aiter_tensor_t> workspace,
    int splitK);
"""
        with open(os.path.join(self.working_path, "opus_gemm_manifest.h"), "w") as f:
            f.write(MANIFEST_HEAD)
            for k in kernels_dict.values():
                if k.kernel_tag.startswith("a8w8_mxscale_bmm_"):
                    f.write(MANIFEST_BMM_MXSCALE.format(kernel_name=k.name))
                elif k.kernel_tag in SPLITK_TAGS:
                    f.write(MANIFEST_A16W16_WORKSPACE.format(kernel_name=k.name))
                elif k.kernel_tag in A16W16_KID_DISPATCH_TAGS:
                    f.write(MANIFEST_A16W16.format(kernel_name=k.name))
                elif k.kernel_tag == "a8w8":
                    f.write(MANIFEST_NOSCALE_3ARG.format(kernel_name=k.name))
                elif k.kernel_tag == "a8w8_scale":
                    f.write(MANIFEST_BLOCKSCALE.format(kernel_name=k.name))
                elif k.kernel_tag in A8W8_BPRESHUFFLE_TAGS:
                    f.write(MANIFEST_BLOCKSCALE_BPRESHUFFLE.format(kernel_name=k.name))
                else:
                    raise ValueError(f"no manifest ABI for kernel tag {k.kernel_tag!r}")

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

        This TU needs no arch guard: it is host-pass only, so a mixed
        build's device passes already see nothing here.
        """

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
            extra_template_params = reduce_abi.get(
                "forward_decl_extra_template_params", ""
            )
            forward_decls = (
                "// Forward declaration only. Specialisations live in per-arch device TUs.\n"
                f"{reduce_abi['forward_decl_include']}"
                "template<int VEC_, int BLOCK_, typename D_OUT,\n"
                "         bool HAS_BIAS_, typename D_BIAS_,\n"
                f"         bool HAS_OOB_{extra_template_params}>\n"
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

        Both the #include and the instantiations sit behind the kid's own
        arch guard, so a mixed build's other device passes see an empty TU
        (see _own_arch_device_pass_guard).
        """
        for row in self._device_instantiations:
            name = row["kid_name"]
            dtype = row["dtype"]
            guard_open, guard_close = _own_arch_device_pass_guard(_kid_name_arch(name))
            # Include the kid's .cuh -- it transitively pulls in the full pipeline header (because
            # OPUS_FUSED_HOST_TU is NOT defined here) an...
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                "// Auto-generated. Do not edit. See gen_instances.py:_emit_device_tus.\n"
                "//\n"
                "// Device-only translation unit for one (kid, dtype) pair.\n"
                "// Keep both JIT and prebuild host passes on the minimal branch --\n"
                "// no torch and no full HIP runtime.\n"
                "#ifndef __HIPCC_RTC__\n"
                "#define __HIPCC_RTC__ 1\n"
                "#endif\n"
                + guard_open
                + f'#include "impl/{name}.cuh"\n'
                + row["device_decl"]
                + guard_close
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
            reduce_header = (
                "gfx942/a16w16/splitk_reduce_gfx942.cuh"
                if reduce_arch == "gfx942"
                else f"{reduce_arch}/splitk_reduce_{reduce_arch}.cuh"
            )
            reduce_abi = SPLITK_REDUCE_ABI_MAP[reduce_arch]
            ws_ptr_type = reduce_abi["ws_type"]
            reduce_kernel = reduce_abi["kernel"]
            if reduce_arch == "gfx1250":
                reduce_vec, reduce_block = 8, 128
                reduce_split_ks = tuple(range(17))
                reduce_workspace_types = ("__bf16", "float")
            else:
                reduce_vec, reduce_block = 16, 64
                reduce_split_ks = (None,)
                reduce_workspace_types = (None,)
            guard_open, guard_close = _own_arch_device_pass_guard(reduce_arch)
            contents = (
                "// SPDX-License-Identifier: MIT\n"
                "// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.\n"
                "//\n"
                f"// Auto-generated per-arch reduce TU ({reduce_arch}). See gen_instances.py:_emit_splitk_reduce_tu.\n"
                "#ifndef __HIPCC_RTC__\n"
                "#define __HIPCC_RTC__ 1\n"
                "#endif\n"
                + guard_open
                + f'#include "{reduce_header}"\n'
                + "".join(
                    _splitk_reduce_baseline_instantiations(
                        reduce_kernel,
                        ws_ptr_type,
                        has_oob,
                        reduce_vec,
                        reduce_block,
                        reduce_split_ks,
                        reduce_workspace_types,
                    )
                    for has_oob in reduce_abi["baseline_has_oob"]
                )
            )
            extra_reduce = SPLITK_REDUCE_EXTRA_MAP.get(reduce_arch, {})
            contents += extra_reduce.get("device_instantiations", lambda: "")()
            contents += guard_close
            Path(
                os.path.join(
                    self.instances_path, f"splitk_reduce_{reduce_arch}.device.cu"
                )
            ).write_text(contents)

    def gen_instances(self, kernels_dict):
        """Regenerate launchers, manifests and exact-kid tables."""
        # A rerun in an existing blob directory must not leave removed or
        # renamed generated policy headers behind.
        for legacy_header in (
            "opus_gemm_lookup.h",
            "opus_gemm_a16w16_tune_lookup.h",
            "opus_gemm_a8w8_tune_lookup.h",
            "opus_bmm_mxscale_tune_lookup.h",
        ):
            Path(self.working_path, legacy_header).unlink(missing_ok=True)

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

        for k in kernels_dict.values():
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

        self.gen_manifest_head(kernels_dict)
        self.gen_a16w16_kid_dispatch(kernels_dict)
        self.gen_a8w8_kid_dispatch(kernels_dict)
        self.gen_bmm_mxscale_kid_dispatch()


def _tune_df_kids(df):
    """Read kid values from either supported tuned-CSV column name."""
    kids = None
    for col in ("solidx", "kernelId"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        kids = values if kids is None else kids.fillna(values)
    return kids


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
        help="generate all kernel instances for tuning (exact-kid dispatch)",
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
            "contribute their `solidx`/`kernelId` only to the subset-compile "
            "set S. Runtime callers provide the final kid explicitly. "
            "Without this flag the module is generated from the sidecar, "
            "per-arch default compile floor, and mandatory family kids."
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
            "contents + DEFAULT_COMPILED_KIDS so subsequent rebuilds "
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
        "gfx942_nosplit": gfx942_nosplit_kernels_list,
        "gfx942_splitk": gfx942_splitk_kernels_list,
        "gfx942_a8w8": gfx942_a8w8_kernels_list,
        "a16w16_cluster_tdm_splitk_ws": gfx1250_kernels_list,
        "a16w16_clusterlaunch_tdm_splitk_ws": gfx1250_clusterlaunch_kernels_list,
        "a16w16_clusterlaunch_tdm_splitk_fuse": gfx1250_splitk_fuse_kernels_list,
        "a16w16_4wave_co": {
            kid: instance
            for kid, instance in gfx1250_4wave_co_kernels_list.items()
            if instance.kernel_tag == "a16w16_4wave_co"
        },
        "a16w16_4wave_wl_co": {
            kid: instance
            for kid, instance in gfx1250_4wave_co_kernels_list.items()
            if instance.kernel_tag == "a16w16_4wave_wl_co"
        },
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
        kids = _tune_df_kids(df)
        if kids is None:
            continue
        for v in kids.dropna().tolist():
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
                sidecar_kids = {int(x) for x in json.load(f)}
        except (OSError, ValueError):
            sidecar_kids = set()

    # The compile set: union, intersected with valid kernels_list entries.
    # MXFP8 BMM launchers are emitted as one gfx950 family below and deduplicated
    # by generated symbol name, so they never participate in the per-kid subset.
    valid_kids = set(kernels_list.keys())
    S = (csv_kids | sidecar_kids | set(DEFAULT_COMPILED_KIDS)) & valid_kids
    S -= set(BMM_MXSCALE_KIDS)

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
        except Exception:  # noqa: BLE001
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
        f.writelines(
            f"#define OPUS_BUILD_HAS_{a.upper()} 1\n" for a in archs_for_header
        )

    # Family ABI defaults must be linkable even when no tuned row or sidecar
    # mentions them.  This set is arch-scoped so single-arch builds never pull
    # another architecture's launcher symbol into their host TU.
    mandatory_arches = (
        set(OPUS_MANDATORY_A8_KIDS) if target_arches is None else set(target_arches)
    )
    mandatory_a8_kids = set().union(
        *(OPUS_MANDATORY_A8_KIDS.get(arch, frozenset()) for arch in mandatory_arches)
    )
    S |= mandatory_a8_kids & valid_kids

    # Honor --kernel_tag as a developer override that *further restricts* the set (within the a16w16
    # / a8w8 families).
    if args.kernel_tag:
        tag_keys = set(TAG_TO_LIST.get(args.kernel_tag, {}).keys())
        if tag_keys:
            # Restrict to the requested family + default compile floor.
            S = (S & tag_keys) | set(default_compiled_kids_for_arch(target_arches))
            S |= mandatory_a8_kids & valid_kids

    # Default exact-id compile-floor invariant (single source of truth:
    # opus_gemm_common.py). C++ and Python both perform exact-kid routing.
    required_default = set(default_compiled_kids_for_arch(target_arches))
    missing_default = required_default - S
    assert not missing_default, (
        f"Subset-compile error: default exact-id kids "
        f"{sorted(missing_default)} are missing from the compile set S. "
        f"Add them to the compile set or update DEFAULT_COMPILED_KIDS "
        f"in csrc/opus_gemm/opus_gemm_common.py."
    )

    # Build the per-kid dict that drives codegen.
    kdict = {kid: kernels_list[kid] for kid in sorted(S)}

    # All 45 BMM ids are exact-routable in the canonical registry.  Several ids
    # intentionally share one device geometry, so key this codegen-only merge by
    # symbol name to emit each host/device specialization once.
    if target_arches is None or "gfx950" in target_arches:
        for family in a8w8_mxscale_bmm_kernel_lists:
            for instance in family.values():
                kdict[instance.name] = instance

    print(
        f"[opus gen_instances] subset compile: |S|={len(S)} kids "
        f"(sources: CSV={len(S & csv_kids)}, "
        f"sidecar={len(S & sidecar_kids)}, "
        f"default-compiled={len(S & required_default)}, "
        f"mandatory-a8={len(S & mandatory_a8_kids)})"
    )

    codegen = opus_gemm_codegen(args.working_path, args.tune)
    codegen.gen_instances(kdict)

    # Persist the expanded compile set so subsequent rebuilds reuse it.
    try:
        os.makedirs(os.path.dirname(sidecar_path) or ".", exist_ok=True)
    except OSError:
        pass
    with open(sidecar_path, "w") as f:
        json.dump(sorted(S), f)
    print(f"[opus gen_instances] wrote sidecar with {len(S)} kids: {sidecar_path}")
