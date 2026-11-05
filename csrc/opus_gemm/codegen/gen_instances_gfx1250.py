# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate gfx1250 OPUS A16W16 launchers."""

import os

from codegen.common import (
    register_arch_map,
    register_emit,
    splitk_workspace_type,
    write_if_changed,
)
from codegen.template_env import render as render_template

# ---------------- gfx1250 arch-override maps ----------------

PIPELINE_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": (
        "gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh"
    ),
    "a16w16_clusterlaunch_tdm_splitk_ws": (
        "gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh"
    ),
    "a16w16_clusterlaunch_tdm_splitk_fuse": (
        "gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh"
    ),
    # CO device bodies are built offline. Point generic codegen lookup at the
    # release-toolchain-safe traits header; the host-only emitter never includes
    # either pipeline header.
    "a16w16_4wave_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_wl_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    # wlr (ring/pin) reuses the wl traits; only its device pipeline differs.
    "a16w16_4wave_wlr_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

TRAITS_HEADER_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_wl_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
    "a16w16_4wave_wlr_co": "gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh",
}

KERNEL_FUNC_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "gemm_a16w16_cluster_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "gemm_a16w16_clusterlaunch_tdm_splitk_ws_kernel_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "gemm_a16w16_splitk_fuse_kernel_gfx1250",
    "a16w16_4wave_co": "gemm_a16w16_4wave_compute_body_gfx1250",
    "a16w16_4wave_wl_co": "gemm_a16w16_4wave_wl_body_gfx1250",
    "a16w16_4wave_wlr_co": "gemm_a16w16_4wave_wlr_body_gfx1250",
}

TRAITS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_cluster_tdm_splitk_ws_traits_gfx1250",
    "a16w16_4wave_co": "opus_a16w16_4wave_compute_traits_gfx1250",
    "a16w16_4wave_wl_co": "opus_a16w16_4wave_wl_traits_gfx1250",
    "a16w16_4wave_wlr_co": "opus_a16w16_4wave_wl_traits_gfx1250",
}

KARGS_NAME_MAP = {
    "a16w16_cluster_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_ws": "opus_gemm_cluster_tdm_ws_kargs_gfx1250",
    "a16w16_clusterlaunch_tdm_splitk_fuse": "opus_gemm_splitk_fuse_kargs_gfx1250",
    "a16w16_4wave_co": "opus_gemm_4wave_compute_kargs_gfx1250",
    "a16w16_4wave_wl_co": "opus_gemm_4wave_compute_kargs_gfx1250",
    "a16w16_4wave_wlr_co": "opus_gemm_4wave_compute_kargs_gfx1250",
}


def co_traits_args(k):
    """Return the traits arguments shared by the offline and host generators."""
    d_a, d_b, d_c, d_acc = k.co_dtypes
    args = (
        f"{k.BLOCK_SIZE}, {k.B_M}, {k.B_N}, {k.B_K}, {k.num_slots}, "
        f"{d_a}, {d_b}, {d_c}, {d_acc}, "
        f"{k.cluster_wg_m}, {k.cluster_wg_n}"
    )
    if k.kernel_tag in ("a16w16_4wave_wl_co", "a16w16_4wave_wlr_co"):
        args += f", {k.co_wave_layout[0]}, {k.co_wave_layout[1]}"
    return args


# fuse workspace storage dtype -> (C type, byte size) for the fuse kernel instantiation.
_FUSE_WS_CTYPE = {"bf16_t": ("__bf16", 2), "fp32_t": ("float", 4)}


def splitk_reduce_extra_device_instantiations():
    # The shared generator emits matched output/bias combinations for both
    # physical workspace types and every split_k specialization. gfx1250 also
    # accepts fp32 bias with bf16 output, so add that mixed combination here.
    out = "// gfx1250 fp32-bias + bf16-output reduce variants\n"
    for has_oob in ("true", "false"):
        for split_k in range(17):
            for workspace_type in ("__bf16", "float"):
                out += (
                    "template __global__ void "
                    "splitk_reduce_kernel_gfx1250<"
                    f"8, 128, __bf16, true, float, {has_oob}, "
                    f"{split_k}, {workspace_type}>(\n"
                    "    const void*, __bf16*, int, int, int, int, int, int,\n"
                    "    const float*, int);\n"
                )
    return out


SPLITK_REDUCE_EXTRA_MAP = {
    "device_instantiations": splitk_reduce_extra_device_instantiations,
}

register_arch_map("gfx1250", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx1250", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx1250", "kernel_func", KERNEL_FUNC_MAP)
register_arch_map("gfx1250", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx1250", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx1250", "splitk_reduce_extra", SPLITK_REDUCE_EXTRA_MAP)

# tileN = consumers split N (B_N>=32); tileM = consumers split M (B_M>=32).
_LAYOUT_INT = {"tileN": 0, "tileM": 1}


# ---------------- gfx1250 emit ----------------


def gen_cluster_tdm_splitk_ws_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    BIAS_HOST_VALIDATE="",
    **_unused,
):
    """Emit a checked gfx1250 two-stage split-K launcher."""
    workspace_dtype, workspace_ptr_type, workspace_aiter_dtype = splitk_workspace_type(
        k
    )
    # The final #4246 reducer uses the same coalesced VEC=8/BLOCK=128 geometry
    # for either physical workspace type.
    reduce_vec, reduce_bs = 8, 128
    layout_int = _LAYOUT_INT[getattr(k, "ctdm_layout", "tileN")]
    has_oob_str = "true" if k.has_oob else "false"
    enable_bias_str = "true" if getattr(k, "enable_bias", False) else "false"

    # CLUSTER-LAUNCH variant: __cluster_dims__(CWM, CWN, 1) multicast TDM. The
    # plain (no-cluster) variant leaves these empty so it is unchanged.
    is_clusterlaunch = k.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_ws"
    cwm = getattr(k, "cluster_wg_m", 4)
    cwn = getattr(k, "cluster_wg_n", 4)

    INSTANCE_IMPL = render_template(
        "gfx1250_cluster_splitk_ws.cuh.j2",
        k=k,
        name=k.name,
        pipeline_header=pipeline_header,
        traits_header=traits_header,
        kernel_func=kernel_func,
        kargs_name=kargs_name,
        traits_name=traits_name,
        da=da,
        db=db,
        workspace_dtype=workspace_dtype,
        workspace_ptr_type=workspace_ptr_type,
        workspace_aiter_dtype=workspace_aiter_dtype,
        layout_int=layout_int,
        has_oob_str=has_oob_str,
        enable_bias_str=enable_bias_str,
        is_clusterlaunch=is_clusterlaunch,
        cluster_wg_m=cwm,
        cluster_wg_n=cwn,
        num_slots=getattr(k, "num_slots", 3),
        wg_per_cu=getattr(k, "wg_per_cu", 2),
        reduce_vec=reduce_vec,
        reduce_bs=reduce_bs,
    )
    write_if_changed(os.path.join(cg.impl_path, f"{k.name}.cuh"), INSTANCE_IMPL)

    # The <fp32_t> token is the host launch-dispatch specialization. The physical
    # workspace type is independently embedded in the Traits alias above.
    for CDtype in k.output_dtypes:
        host_decl = (
            f"template void\n"
            f"{k.name}<{CDtype}>(\n"
            f"    aiter_tensor_t &XQ,\n"
            f"    aiter_tensor_t &WQ,\n"
            f"    aiter_tensor_t &Y,\n"
            f"    aiter_tensor_t &workspace,\n"
            f"    std::optional<aiter_tensor_t>,\n"
            f"    int);\n"
        )
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<{CDtype}>>({kargs_name});\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_decl": host_decl}
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "device_decl": device_decl}
        )


def gen_splitk_fuse_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    BIAS_HOST_VALIDATE="",
    **_unused,
):
    """Emit the fused gfx1250 split-K launcher and workspace checks."""
    del BIAS_HOST_VALIDATE
    workspace_dtype, workspace_ptr_type, workspace_aiter_dtype = splitk_workspace_type(
        k
    )
    layout_int = _LAYOUT_INT[getattr(k, "ctdm_layout", "tileN")]
    enable_bias_str = "true" if getattr(k, "enable_bias", False) else "false"
    num_slots = getattr(k, "num_slots", 3)
    wg_per_cu = getattr(k, "wg_per_cu", 2)
    split_k = int(k.fuse_split_k)
    # Historical field name retained for compatibility; physically cluster.y
    # groups N-tile peers sharing A.
    n_cluster = int(k.fuse_m_cluster)
    if split_k < 2:
        raise ValueError(f"fused instance {k.name} must declare fuse_split_k >= 2")

    INSTANCE_IMPL = render_template(
        "gfx1250_splitk_fuse.cuh.j2",
        k=k,
        name=k.name,
        pipeline_header=pipeline_header,
        traits_header=traits_header,
        kernel_func=kernel_func,
        kargs_name=kargs_name,
        traits_name=traits_name,
        da=da,
        db=db,
        workspace_dtype=workspace_dtype,
        workspace_ptr_type=workspace_ptr_type,
        workspace_aiter_dtype=workspace_aiter_dtype,
        layout_int=layout_int,
        enable_bias_str=enable_bias_str,
        num_slots=num_slots,
        wg_per_cu=wg_per_cu,
        split_k=split_k,
        split_k_minus_1=split_k - 1,
        n_cluster=n_cluster,
    )
    write_if_changed(os.path.join(cg.impl_path, f"{k.name}.cuh"), INSTANCE_IMPL)

    host_decl = (
        f"template void\n"
        f"{k.name}<fp32_t>(\n"
        f"    aiter_tensor_t &XQ,\n"
        f"    aiter_tensor_t &WQ,\n"
        f"    aiter_tensor_t &Y,\n"
        f"    aiter_tensor_t &workspace,\n"
        f"    std::optional<aiter_tensor_t>,\n"
        f"    int);\n"
    )
    cg._host_instantiations.append(
        {"kid_name": k.name, "dtype": "fp32_t", "host_decl": host_decl}
    )
    for d_out in ("__bf16", "float"):
        device_decl = (
            f"template __global__ void {kernel_func}<\n"
            f"    {k.name}_Traits<fp32_t>, {split_k}, {workspace_ptr_type}, "
            f"{n_cluster}, {d_out}>({kargs_name});\n"
        )
        cg._device_instantiations.append(
            {"kid_name": k.name, "dtype": d_out, "device_decl": device_decl}
        )


def gen_4wave_co_instance(
    cg,
    k,
    traits_header,
    traits_name,
    kargs_name,
    **_unused,
):
    """Emit a host launcher for one pre-built gfx1250 CO exact kid.

    The launcher uses the repository's existing five-argument non-workspace
    A16W16 ABI. No device instantiation is emitted: the matching device image is
    selected from the regular or A0 subtree on first use per device/image path,
    with the same exact kid and symbol on both steppings.
    """
    instance_impl = render_template(
        "gfx1250_4wave_co.cuh.j2",
        name=k.name,
        traits_header=traits_header,
        traits_name=traits_name,
        co_traits_args=co_traits_args(k),
        kargs_name=kargs_name,
    )
    write_if_changed(os.path.join(cg.impl_path, f"{k.name}.cuh"), instance_impl)

    # Intentionally append host instantiations only. This is the compile bypass
    # that keeps release hipcc away from the pin-VGPR device pipeline.
    for c_dtype in k.output_dtypes:
        host_decl = (
            "template void\n"
            f"{k.name}<{c_dtype}>(\n"
            "    aiter_tensor_t &XQ,\n"
            "    aiter_tensor_t &WQ,\n"
            "    aiter_tensor_t &Y,\n"
            "    std::optional<aiter_tensor_t>,\n"
            "    int);\n"
        )
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": c_dtype, "host_decl": host_decl}
        )


# ---------- Self-register at import time ----------
register_emit("gfx1250", "a16w16_4wave_co", gen_4wave_co_instance)
register_emit("gfx1250", "a16w16_4wave_wl_co", gen_4wave_co_instance)
register_emit("gfx1250", "a16w16_4wave_wlr_co", gen_4wave_co_instance)
register_emit(
    "gfx1250", "a16w16_cluster_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
register_emit(
    "gfx1250", "a16w16_clusterlaunch_tdm_splitk_fuse", gen_splitk_fuse_instance
)
# CLUSTER-LAUNCH variant shares the same emit (it branches on k.kernel_tag to add
# __cluster_dims__, physical-grid round-up, and the CLUSTER_WG_M/N traits args).
register_emit(
    "gfx1250", "a16w16_clusterlaunch_tdm_splitk_ws", gen_cluster_tdm_splitk_ws_instance
)
