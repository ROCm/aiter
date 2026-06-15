# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 codegen -- emit launchers for gfx942-targeted kid families."""

import os
from pathlib import Path

from opus_gemm_common import OpusGemmInstance

from codegen.common import (
    WARP_SIZE,
    _GFX942_A16W16_TAGS,
    _NOSPLIT,
    _SPLITK,
    W3_KERNEL_PAIRS,
    register_arch_map,
    register_emit,
)
from codegen.template_env import render as _render


# gfx942 pipeline header derived from W3_KERNEL_PAIRS: splitk_X reuses
# nosplit_X's .cuh (paired template); splitk_fused has its own.
def _gfx942_pipeline(tag):
    return f"gfx942/opus_gemm_pipeline_{tag}.cuh"


# Traits header carries the traits struct + kargs struct definitions for a given pipeline tag.
GFX942_TRAITS_HEADER = "gfx942/opus_gemm_traits_a16w16.cuh"

# gfx942 a16w16 tags all share one traits class name (no arch suffix).
GFX942_TRAITS_NAME = "opus_gemm_a16w16_traits"

# gfx942 a16w16 family supports only the 16x16x16 BF16 MFMA shape.
VALID_GFX942_BF16_MFMA = {(16, 16, 16)}

PIPELINE_HEADER_MAP = {
    "a16w16_fused_reduce": _gfx942_pipeline("a16w16_fused_reduce"),
    "a16w16_kbuf1_large_tile": _gfx942_pipeline("a16w16_kbuf1_large_tile"),
    **{nosplit: _gfx942_pipeline(nosplit) for nosplit in _NOSPLIT},
    **{
        splitk: _gfx942_pipeline(nosplit) for nosplit, splitk in W3_KERNEL_PAIRS.items()
    },
}

TRAITS_HEADER_MAP = {tag: GFX942_TRAITS_HEADER for tag in _GFX942_A16W16_TAGS}

TRAITS_NAME_MAP = {tag: GFX942_TRAITS_NAME for tag in _GFX942_A16W16_TAGS}

KARGS_NAME_MAP = {
    "a16w16_fused_reduce": "opus_gemm_splitk_fused_kargs",
    "a16w16_kbuf1_large_tile": "opus_gemm_noscale_kargs",
    **{tag: "opus_gemm_splitk_kargs" for tag in _SPLITK},
    **{tag: "opus_gemm_noscale_kargs" for tag in _NOSPLIT},
}

KERNEL_FUNC_MAP = {
    "a16w16_fused_reduce": "gemm_a16w16_fused_reduce_kernel",
    "a16w16_kbuf1_large_tile": "gemm_a16w16_kbuf1_large_tile_kernel",
    # gfx942 paired tags: nosplit_tag's kernel symbol; splitk_tag reuses it.
    **{nosplit: f"gemm_{nosplit}_kernel" for nosplit in W3_KERNEL_PAIRS.keys()},
    **{splitk: f"gemm_{nosplit}_kernel" for nosplit, splitk in W3_KERNEL_PAIRS.items()},
}

register_arch_map("gfx942", "pipeline_header", PIPELINE_HEADER_MAP)
register_arch_map("gfx942", "traits_header", TRAITS_HEADER_MAP)
register_arch_map("gfx942", "traits_name", TRAITS_NAME_MAP)
register_arch_map("gfx942", "kargs_name", KARGS_NAME_MAP)
register_arch_map("gfx942", "kernel_func", KERNEL_FUNC_MAP)


def gen_splitk_gfx942_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    fused,
    kargs_template_vars,
    SPLITK_REDUCE_FAST_ARCHES,
    V3_NVEC_ROWS,
    V2_SUPPORTED_SPLITKS,
    A16W16_TUNE_HOST_EXTRA,
    record_one_instantiation,
    **_unused,
):
    """gfx942 a16w16 splitk launcher emit (fused / non-fused)."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )

    # gfx942 a16w16_traits: 6 params <BLOCK_SIZE, BLOCK, DTYPE, VEC, TILE, WAVE>.
    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, fp32_t, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>,
    opus::seq<{k.T_M}, {k.T_N}, 1>,
    opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>>;
"""

    # Per-flavor pieces (workspace alloc + reduce dispatch).
    if fused:
        err_label = "a16w16_fused_reduce"
        ws_alloc_extra = """
    int num_flags = batch * num_tiles_m * num_tiles_n;
    size_t total_bytes = ws_bytes + (size_t)num_flags * sizeof(unsigned int);"""
        ws_size_var = "total_bytes"
        flags_block = """
    unsigned int* ptr_flags_ = reinterpret_cast<unsigned int*>(
    static_cast<char*>(ws_cached_ptr) + ws_bytes);
    HIP_CALL(hipMemsetAsync(ptr_flags_, 0, num_flags * sizeof(unsigned int), stream));"""
        kargs_flags_assign = "    kargs.ptr_flags     = ptr_flags_;\n"
        # cooperative_reduce: 1 if all split_k WGs co-resident (split reduce work), else 0.
        cooperative_assign = """
    static thread_local int cu_for_coop = -1;
    if (cu_for_coop < 0) {{
    int dev_c = 0;
    hipDeviceProp_t prop_c{{}};
    if (hipGetDevice(&dev_c) == hipSuccess &&
        hipGetDeviceProperties(&prop_c, dev_c) == hipSuccess) {{
        cu_for_coop = prop_c.multiProcessorCount;
    }}
    if (cu_for_coop <= 0) cu_for_coop = 64;
    }}
    int total_wgs_coop = num_tiles_m * num_tiles_n * batch * split_k;
    kargs.cooperative_reduce = (total_wgs_coop <= cu_for_coop) ? 1 : 0;
"""
        # fused: D_OUT template param so in-kernel reduce casts to Y.dtype() (avoid bf16/fp32 mismatch).
        kernel_fwd_decl = (
            f"template<typename Traits, typename D_OUT>\n"
            f"__global__ void {kernel_func}({kargs_name} kargs);"
        )
        kernel_launch_body = f"""
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
    {kernel_func}<{k.name}_Traits<D_C>, __bf16><<<grid_main, block_main, 0, stream>>>(kargs);
    }}}} else {{{{
    {kernel_func}<{k.name}_Traits<D_C>, float><<<grid_main, block_main, 0, stream>>>(kargs);
    }}}}"""
        reduce_launch = ""  # in-kernel reduce; no separate launch
    else:
        err_label = k.kernel_tag
        ws_alloc_extra = ""
        ws_size_var = "ws_bytes"
        flags_block = ""
        kargs_flags_assign = ""
        cooperative_assign = ""
        # non-fused splitk: D_C only; Y-dtype dispatch inside separate reduce kernel.
        kernel_fwd_decl = (
            f"template<typename Traits{fwd_decl_kargs_tpl}>\n"
            f"__global__ void {kernel_func}({fwd_decl_kargs_fnarg} kargs);"
        )
        # Kargs deduced from kargs fn arg; <Traits> only keeps host/device
        # mangling identical (avoid SA_ vs T0_ substitution mismatch).
        kernel_launch_body = (
            f"\n    {kernel_func}<{k.name}_Traits<D_C>>"
            f"<<<grid_main, block_main, 0, stream>>>(kargs);"
        )
        # V2 essential for N=64+M%row!=0 (V3 misses); baseline 50-80% slower.
        v2_enabled = k.arch_prefix in SPLITK_REDUCE_FAST_ARCHES
        v2_prelude = (
            """
    // V2/V3 fast path: split_k static-unroll, no OOB.
    constexpr int V2_VEC = 8;
    constexpr int V2_BS  = 8;
    const bool v2_align = (N % (V2_VEC * V2_BS) == 0) && (padded_N == N);
    dim3 grid_reduce_v2(v2_align ? (N / (V2_VEC * V2_BS)) : 1, batch * M, 1);
    dim3 block_reduce_v2(V2_BS);

    // V3: multi-row per wg (BLOCK = N_VEC * ROWS_PER_BLOCK = 64, 1 full wave).
    // Dispatch picks the (N_VEC, ROWS) tuple at runtime; supported set is in
    // V3_NVEC_ROWS (gen_instances.py).
    const int v3_n_vec = N / V2_VEC;
"""
            if v2_enabled
            else ""
        )

        def v2_branch(hasbias):
            if not v2_enabled:
                return ""
            hb = "true" if hasbias else "false"
            bias_arg = (
                "reinterpret_cast<const __bf16*>(ptr_bias_), stride_bias_batch_"
                if hasbias
                else "nullptr, 0"
            )
            # V3 branches first; V2 fallback when no V3 (N_VEC, ROWS) tuple matches.
            branches = []
            first = True
            for nvec, rows in V3_NVEC_ROWS:
                block_size = nvec * rows
                for sk in V2_SUPPORTED_SPLITKS:
                    kw = "if" if first else "else if"
                    first = False
                    branches.append(
                        f"""            {kw} (v2_align && v3_n_vec == {nvec} && (M % {rows} == 0) && split_k == {sk}) {{{{{{{{
            dim3 grid_v3(1, M / {rows}, batch);
            dim3 block_v3({block_size});
            splitk_reduce_kernel_v3<{sk}, {nvec}, {rows}, V2_VEC, __bf16, {{hb}}, __bf16>
                <<<grid_v3, block_v3, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    M, N, batch, padded_M, padded_N,
                    {{bias_arg}});
        }}}}}}}}""".format(
                            hb=hb, bias_arg=bias_arg
                        )
                    )
            for sk in V2_SUPPORTED_SPLITKS:
                branches.append(
                    f"""            else if (v2_align && split_k == {sk}) {{{{{{{{
            splitk_reduce_kernel_v2<{sk}, V2_VEC, V2_BS, __bf16, {{hb}}, __bf16>
                <<<grid_reduce_v2, block_reduce_v2, 0, stream>>>(
                    reinterpret_cast<const float*>(ptr_workspace_),
                    reinterpret_cast<__bf16*>(Y.data_ptr()),
                    M, N, batch, padded_M, padded_N,
                    {{bias_arg}});
        }}}}}}}}""".format(hb=hb, bias_arg=bias_arg)
                )
            return "\n".join(branches) + " else "  # falls through to baseline

        # Baseline reduce call (V2/V3 fall through here; fp32 always lands here).
        def _baseline_call(dtype, hasbias, indent):
            hb = "true" if hasbias else "false"
            bias_args = (
                f"\n{indent}            reinterpret_cast<const {dtype}*>(ptr_bias_),\n"
                f"{indent}            stride_bias_batch_);"
                if hasbias
                else f"\n{indent}            nullptr, 0);"
            )
            return (
                f"{indent}splitk_reduce_kernel<REDUCE_VEC, REDUCE_BS, {dtype}, {hb}, {dtype}, true>\n"
                f"{indent}    <<<grid_reduce, block_reduce, 0, stream>>>(\n"
                f"{indent}        reinterpret_cast<const float*>(ptr_workspace_),\n"
                f"{indent}        reinterpret_cast<{dtype}*>(Y.data_ptr()),\n"
                f"{indent}        split_k, M, N, batch, padded_M, padded_N,"
                f"{bias_args}"
            )

        bf16_t = _baseline_call("__bf16", True, "                ")
        bf16_f = _baseline_call("__bf16", False, "                ")
        fp32_t = _baseline_call("float", True, "            ")
        fp32_f = _baseline_call("float", False, "            ")
        reduce_launch = f"""
    constexpr int REDUCE_VEC = 16;
    constexpr int REDUCE_BS  = 64;
    dim3 grid_reduce((N + REDUCE_VEC * REDUCE_BS - 1) / (REDUCE_VEC * REDUCE_BS),
                  batch * M, 1);
    dim3 block_reduce(REDUCE_BS);
{v2_prelude}
    if (Y.dtype() == AITER_DTYPE_bf16) {{{{
    if (bias.has_value()) {{{{
{v2_branch(True)}{{{{
{bf16_t}
        }}}}
    }}}} else {{{{
{v2_branch(False)}{{{{
{bf16_f}
        }}}}
    }}}}
    }}}} else {{{{
    // fp32 output: V2 not implemented yet; use baseline.
    if (bias.has_value()) {{{{
{fp32_t}
    }}}} else {{{{
{fp32_f}
    }}}}
    }}}}"""

    target_wg_expr = "2 * cu_cached" if k.kernel_tag.endswith("_p1") else "cu_cached"
    require_even_loops_str = (
        "true"
        if k.kernel_tag in ("a16w16_kbuf2v_sk", "a16w16_kbuf2v_bk128_sk")
        else "false"
    )
    INSTANCE_IMPL = _render(
        "impl_splitk_gfx942.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        kernel_fwd_decl=kernel_fwd_decl,
        traits_aliases=traits_aliases,
        kid_name=k.name,
        kargs_name=kargs_name,
        err_label=err_label,
        B_K=k.B_K, B_M=k.B_M, B_N=k.B_N,
        BLOCK_SIZE=k.BLOCK_SIZE,
        target_wg_expr=target_wg_expr,
        require_even_loops_str=require_even_loops_str,
        ws_alloc_extra=ws_alloc_extra,
        ws_size_var=ws_size_var,
        flags_block=flags_block,
        kargs_flags_assign=kargs_flags_assign,
        cooperative_assign=cooperative_assign,
        kernel_launch_body=kernel_launch_body,
        reduce_launch=reduce_launch,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    if fused:
        # fused: instantiate both Y dtypes (bf16, float) so runtime dispatch links.
        for CDtype in k.output_dtypes:
            cg._host_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "host_extra_params": A16W16_TUNE_HOST_EXTRA,
                }
            )
            # fused kernel has a second D_OUT template param for in-kernel Y-dtype dispatch.
            cg._device_instantiations.append(
                {
                    "kid_name": k.name,
                    "dtype": CDtype,
                    "kernel_func": kernel_func,
                    "kargs_name": kargs_name,
                    "kargs_explicit_param": ", __bf16",
                    "extra_device_decls": [
                        {
                            "kargs_explicit_param": ", float",
                        }
                    ],
                }
            )
    else:
        record_one_instantiation(
            cg,
            k,
            kernel_func,
            kargs_name,
            A16W16_TUNE_HOST_EXTRA,
            kargs_explicit_param,
        )


def gen_a16w16_nosplit_gfx942_instance(
    cg,
    k,
    pipeline_header,
    traits_header,
    kernel_func,
    da,
    db,
    traits_name,
    kargs_name,
    kargs_template_vars,
    record_one_instantiation,
    A16W16_TUNE_HOST_EXTRA,
    A16W16_TUNE_TAGS,
    **_unused,
):
    """gfx942 a16w16 non-splitK launcher emit (kbuf1_large_tile / kbuf3 / kbuf2v /
    kbuf2v_bk128 / kbuf1). Single Traits, no bias, no cachectl, no HAS_OOB tail."""
    kargs_explicit_param, fwd_decl_kargs_tpl, fwd_decl_kargs_fnarg = (
        kargs_template_vars(k.kernel_tag, kargs_name)
    )
    traits_extra = (
        f",\n        opus::seq<{k.T_M}, {k.T_N}, 1>,"
        f"\n        opus::seq<{k.W_M}, {k.W_N}, {k.W_K}>"
    )

    min_k = 2 * k.B_K
    k_check = f"""
    int loops_ = (K + {k.B_K} - 1) / {k.B_K};
    AITER_CHECK(loops_ >= 2,
        "K=", K, " too small for B_K={k.B_K}, need K >= {min_k}");
    AITER_CHECK(loops_ % 2 == 0,
        "ceil_div(K, {k.B_K})=", loops_, " must be even (prefetch constraint)");
    AITER_CHECK(K % 2 == 0,
        "K=", K, " must be even (a16w16 family rejects odd K due to a "
        "latent K-tail accumulation bug; pass an even K)");
    AITER_CHECK(M >= 1 && N >= 1, "M and N must be >= 1");
"""

    extra_param = (
        ",\n    std::optional<aiter_tensor_t> bias," "\n    int /*splitK*/"
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )

    bias_kargs_block = (
        "    AITER_CHECK(!bias.has_value(),\n"
        '        "bias not supported on this a16w16 kid");\n'
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )

    traits_aliases = f"""
template <typename D_C>
using {k.name}_Traits = {traits_name}<{k.BLOCK_SIZE},
    opus::seq<{k.B_M}, {k.B_N}, {k.B_K}>,
    opus::tuple<{da}, {db}, D_C, fp32_t>,
    opus::seq<{k.VEC_A}, {k.VEC_B}, {k.VEC_C}>{traits_extra}>;
"""

    launch_block = f"""
    auto stream = aiter::getCurrentHIPStream();
    {kernel_func}<{k.name}_Traits<D_C>><<<grid, block, 0, stream>>>(kargs);"""

    INSTANCE_IMPL = _render(
        "impl_nosplit_gfx942.cuh.j2",
        traits_header=traits_header,
        pipeline_header=pipeline_header,
        fwd_decl_kargs_tpl=fwd_decl_kargs_tpl,
        fwd_decl_kargs_fnarg=fwd_decl_kargs_fnarg,
        kernel_func=kernel_func,
        traits_aliases=traits_aliases,
        kid_name=k.name,
        kargs_name=kargs_name,
        extra_param=extra_param,
        k_check=k_check,
        bias_kargs_block=bias_kargs_block,
        B_M=k.B_M, B_N=k.B_N,
        BLOCK_SIZE=k.BLOCK_SIZE,
        launch_block=launch_block,
    )
    Path(os.path.join(cg.impl_path, f"{k.name}.cuh")).write_text(INSTANCE_IMPL)

    inst_extra_param = (
        ",\n    std::optional<aiter_tensor_t>,\n    int"
        if k.kernel_tag in A16W16_TUNE_TAGS
        else ""
    )
    for CDtype in k.output_dtypes:
        cg._host_instantiations.append(
            {"kid_name": k.name, "dtype": CDtype, "host_extra_params": inst_extra_param}
        )
        cg._device_instantiations.append(
            {
                "kid_name": k.name,
                "dtype": CDtype,
                "kernel_func": kernel_func,
                "kargs_name": kargs_name,
                "kargs_explicit_param": kargs_explicit_param,
            }
        )


# ---------- Self-register at import time ----------
# gfx942 splitk family: 5 tags.
_GFX942_SPLITK_TAGS = (
    "a16w16_kbuf3_sk",
    "a16w16_kbuf2v_sk",
    "a16w16_kbuf2v_bk128_sk",
    "a16w16_kbuf1_sk",
    "a16w16_fused_reduce",
)
for _tag in _GFX942_SPLITK_TAGS:
    register_emit("gfx942", _tag, gen_splitk_gfx942_instance)

# gfx942 a16w16 non-splitK family: 5 tags.
_GFX942_NOSPLIT_TAGS = (
    "a16w16_kbuf1_large_tile",
    "a16w16_kbuf3",
    "a16w16_kbuf2v",
    "a16w16_kbuf2v_bk128",
    "a16w16_kbuf1",
)
for _tag in _GFX942_NOSPLIT_TAGS:
    register_emit("gfx942", _tag, gen_a16w16_nosplit_gfx942_instance)


# ---------------- gfx942 a16w16 validator ----------------
# Coverage: basic physical limits only. Detailed LDS depth / layout checks
# live in gfx942/opus_gemm_traits_a16w16.cuh static_asserts (hipcc enforces).

# gfx942 (CDNA3 / MI300X) hardware LDS budget per WG.
_GFX942_LDS_PER_WG_BYTES = 64 * 1024


def _validate_a16w16_gfx942(k: OpusGemmInstance):
    """Validate a gfx942 a16w16 instance -- basic physical limits only."""
    errors = []

    # MFMA shape: gfx942 a16w16 family is locked to 16x16x16 BF16.
    if (k.W_M, k.W_N, k.W_K) not in VALID_GFX942_BF16_MFMA:
        errors.append(f"WAVE=({k.W_M},{k.W_N},{k.W_K}) not in {VALID_GFX942_BF16_MFMA}")

    # BLOCK_SIZE physical cap (hardware: 1024 max; gfx942 a16w16 we cap at 512).
    if k.BLOCK_SIZE > 512:
        errors.append(f"BLOCK_SIZE={k.BLOCK_SIZE} exceeds 512")

    # AGPR/VGPR register-file caps (hardware: 256 each, 512 combined).
    E_M = (k.B_M // 2) // (k.W_M * k.T_M) if (k.W_M * k.T_M) else 0
    E_N = (k.B_N // 2) // (k.W_N * k.T_N) if (k.W_N * k.T_N) else 0
    E_K = k.B_K // k.W_K if k.W_K else 0
    agpr_per_mfma = (k.W_M * k.W_N) // WARP_SIZE
    total_agprs = 4 * E_M * E_N * agpr_per_mfma
    vgpr_est = 4 * E_K * (E_M + 2 * E_N) + 80
    if total_agprs >= 256:
        errors.append(f"AGPR={total_agprs} must be < 256")
    if vgpr_est > 256:
        errors.append(f"VGPR_est={vgpr_est} exceeds 256")
    if vgpr_est + total_agprs > 512:
        errors.append(f"VGPR+AGPR={vgpr_est + total_agprs} exceeds 512")

    # Loose LDS bound: 2 * B_M * B_K + 2 * B_N * B_K bytes for bf16 (1-deep
    # per slot, ignores pipeline depth + padding). Anything past 64 KiB is
    # physically impossible; finer-grained checks live in traits.cuh.
    lds_min_bytes = 2 * (k.B_M + k.B_N) * k.B_K
    if lds_min_bytes > _GFX942_LDS_PER_WG_BYTES:
        errors.append(
            f"LDS lower bound={lds_min_bytes // 1024}KiB exceeds "
            f"{_GFX942_LDS_PER_WG_BYTES // 1024}KiB (gfx942 budget)"
        )

    if errors:
        msg = f"Invalid gfx942 a16w16 instance '{k.name}':\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return {
        "E_M": E_M,
        "E_N": E_N,
        "E_K": E_K,
        "agprs": total_agprs,
        "vgpr_est": vgpr_est,
        "lds_bytes": lds_min_bytes,
        "min_k": 2 * k.B_K,
    }
