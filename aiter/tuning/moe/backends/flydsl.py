# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL candidates of the fused MoE tuner's two-stage path."""

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    get_flydsl_activation_name,
)
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_stage1_kernels,
    get_flydsl_stage1_kernels_a16w16,
    get_flydsl_stage2_kernels,
    get_flydsl_stage2_kernels_a16w16,
)
from aiter.test_common import cosine_diff_compare
from aiter.tuning.moe.compare import (
    _a16w_sorted_cos,
    run_torch_moe_stage1,
    run_torch_moe_stage2,
)
from aiter.tuning.moe.inputs import generate_data_2stages
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    v2_stage1_sorted_ref as _v2_stage1_ref,
)


def run_flydsl_stage1_out(
    a1_qt,
    w1_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w1_scale_aiter,
    a1_scale,
    bias,
    dtype,
    topk,
    kparams,
    blockM,
    q_dtype_a,
    q_type,
    act_type,
):
    a_scale_one = kparams.get("a_scale_one", False)
    _out_dtype = kparams["out_dtype"]
    token_num = a1_qt.shape[0]
    inter_dim = w1_qt_shffle_ck.shape[1] // 2
    result = flydsl_moe_stage1(
        a=a1_qt.to(dtypes.fp8) if q_dtype_a == dtypes.fp8 else a1_qt,
        w1=w1_qt_shffle_ck,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=kparams["tile_m"],
        tile_n=kparams["tile_n"],
        tile_k=kparams["tile_k"],
        a_dtype=kparams["a_dtype"],
        b_dtype=kparams["b_dtype"],
        out_dtype=_out_dtype,
        act=get_flydsl_activation_name(act_type),
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
        w1_scale=w1_scale_aiter,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        use_async_copy=True,
        k_batch=kparams.get("k_batch", 1),
        waves_per_eu=kparams.get("waves_per_eu", 3),
        b_nt=kparams.get("b_nt", 2),
        gate_mode=kparams.get("gate_mode", "separated"),
        a_scale_one=a_scale_one,
        xcd_swizzle=kparams.get("xcd_swizzle", 0),
        bias=bias,
        k_wave=kparams.get("k_wave", 1),
    )
    if isinstance(result, tuple):
        out_raw = result[0]
        if _out_dtype == "fp4":
            total_fp4_bytes = token_num * topk * (inter_dim // 2)
            fp4_flat = out_raw.view(-1).view(torch.uint8)[:total_fp4_bytes]
            return fp4_flat.view(dtypes.fp4x2).reshape(token_num, topk, -1)
        else:
            # fuse_fp8: out_raw is fp8 tensor, shape (token_num, topk, inter_dim)
            return out_raw.reshape(token_num, topk, -1)
    # a16w-mix (bf16/fp16 activation) returns the SORTED [sorted_size, inter] bf16
    # intermediate as-is (kernel-only, so the tuned latency is clean); it is
    # compared in sorted space against run_a16w_stage1_sorted_ref.
    return result


def run_flydsl_stage2_out(
    a2_qt,
    w2_shuffled_flydsl,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w2_scale_shuffled_flydsl,
    a2_scale,
    moe_buf,
    bias,
    dtype,
    topk,
    kparams,
    blockM,
    q_type,
    act_type,
):
    # a16w4 (bf16 A x mxfp4 W) stage2 always atomic-scatters into moe_buf,
    # whatever the kernelName's mode (same rule as fused_moe's
    # stage2_uses_route_reduce), so it needs the zeroed buffer too.
    is_a16w4 = kparams["a_dtype"] == "bf16" and kparams["b_dtype"] == "fp4"
    if kparams.get("mode", "atomic") == "atomic" or is_a16w4:
        moe_buf.zero_()

    sort_block_m = kparams.get("sort_block_m", 0)
    persist = kparams.get("persist", None)

    # a16w-mix (bf16 activation) stage2 consumes a SORTED bf16 intermediate; the
    # tuner feeds the pre-sorted a2 ("a2_a16w_sorted", built once in generate_data,
    # not per timed iter) and no inter-stage act scale.
    if kparams["a_dtype"] == "bf16":
        a2_scale = None

    return flydsl_moe_stage2(
        inter_states=a2_qt,
        w2=w2_shuffled_flydsl,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=moe_buf,
        topk=topk,
        tile_m=kparams["tile_m"],
        tile_n=kparams["tile_n"],
        tile_k=kparams["tile_k"],
        a_dtype=kparams["a_dtype"],
        b_dtype=kparams["b_dtype"],
        out_dtype=kparams["out_dtype"],
        mode=kparams.get("mode", "atomic"),
        w2_scale=w2_scale_shuffled_flydsl,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=sort_block_m,
        persist=persist,
        b_nt=kparams.get("b_nt", 0),
        xcd_swizzle=kparams.get("xcd_swizzle", 0),
        bias=bias,
    )


def run_a16w_stage1_sorted_ref(
    a1_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    w1_scale,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    dtype,
    act_type,
    quant_type,
    doweight_stage1,
    topk,
    blockM,
    inter_dim,
):
    """a16w-mix stage1 reference in SORTED [max_sorted, inter] bf16 layout, to
    compare against the kernel's sorted output directly (so run_flydsl_stage1_out
    stays kernel-only and its timing isn't polluted by a re-sort). Computes the
    unsorted torch ref then gathers it to sorted order via the expert-match rule.
    """
    ref1 = run_torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        a1_scale=None,
        w1_scale=w1_scale,
        dtype=dtype,
        activation=act_type,
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        topk=topk,
    )
    n = int(num_valid_ids.reshape(-1)[0].item())
    return _v2_stage1_ref(
        ref1.view(a1_qt.shape[0], topk, inter_dim),
        topk_ids,
        sorted_ids,
        sorted_expert_ids,
        n,
        token=a1_qt.shape[0],
        inter_dim=inter_dim,
        bm_s1=blockM,
        max_sorted=sorted_ids.numel(),
    )


def gen_flydsl_2stages_task(info, blockMs):
    tasks_flydsl = []
    (
        _gfx,
        _cu_num,
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        act_type,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        doweight_stage1,
    ) = info

    # a16w16: bf16 A x bf16 W on the a16w-mix port, the kernel the runtime
    # selects for BF16 QuantType.No when swiglu_limit is active.
    # Arch list matches fused_moe._A16WMIX_BF16_ARCHS: the port's
    # non-gfx942 path is gfx950 MFMA; gfx90a does not compile.
    is_a16w16 = (
        q_type == QuantType.No
        and dtype == dtypes.bf16
        and q_dtype_a == dtypes.bf16
        and q_dtype_w == dtypes.bf16
        and act_type in (ActivationType.Silu, ActivationType.Swiglu)
        and not doweight_stage1
        and str(_gfx) in ("gfx942", "gfx950")
    )
    if not is_a16w16 and (
        q_type != QuantType.per_1x32 or q_dtype_w not in (dtypes.fp4x2, dtypes.fp8)
    ):
        return tasks_flydsl

    _a_dtype_map = {
        dtypes.fp8: "fp8",
        dtypes.fp4x2: "fp4",
        dtypes.fp16: "fp16",
        # a16w4: bf16 activation -> "bf16" so we enumerate the abf16_wfp4
        # kernels the runtime actually selects (see fused_moe heuristic
        # fallback kn1/kn2 = flydsl_moe{1,2}_abf16_wfp4_*). Mapping bf16 to
        # "fp16" here would tune afp16_* kernels that never match at runtime.
        dtypes.bf16: "bf16",
    }
    a_dtype_str = _a_dtype_map.get(q_dtype_a, "fp8")
    b_dtype_str = "fp8" if q_dtype_w == dtypes.fp8 else "fp4"
    out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"

    if is_a16w16:
        b_dtype_str = "bf16"
        # The a16w-mix port has no f16 output. AOT replay
        # (aiter.aot.flydsl.moe._caller_inputs) only has a QuantType.No
        # bf16×bf16 recipe; f16 names would break PREBUILD_KERNELS=1.
        out_dtype_str = "bf16"
        # The runtime wrapper passes waves_per_eu=None for registry entries
        # without the key; run_flydsl_stage1_out would otherwise time wpe=3.
        flydsl_s1_kernels = {
            k: {**v, "waves_per_eu": None}
            for k, v in get_flydsl_stage1_kernels_a16w16(out_dtype_str).items()
        }
        flydsl_s2_kernels = get_flydsl_stage2_kernels_a16w16(out_dtype_str)
    else:
        flydsl_s1_kernels = get_flydsl_stage1_kernels(
            a_dtype_str, b_dtype_str, out_dtype_str
        )
        flydsl_s2_kernels = get_flydsl_stage2_kernels(
            a_dtype_str, b_dtype_str, out_dtype_str
        )

    for blockM in blockMs:
        # 16 included: a16w4/a8w4 emit t16 names, s1_tile_m drops the rest.
        if blockM not in [16, 32, 64, 128] or not use_g1u1:
            continue
        for kname, kparams in flydsl_s1_kernels.items():
            is_splitk = kparams.get("k_batch", 1) > 1

            # Drop k_batch/k_wave splits that do not divide the K axis.
            # compile_mixed_moe_gemm1 rejects them, but only after the
            # candidate has been dispatched, and on some shapes the launch
            # faults the queue (HSA_STATUS_ERROR_EXCEPTION) instead of
            # raising, taking the worker pool down with it.
            _kb = kparams.get("k_batch", 1)
            _kw = kparams.get("k_wave", 1)
            _tk = kparams["tile_k"]
            if model_dim % _kb != 0:
                continue
            _k_per_batch = model_dim // _kb
            if _k_per_batch % _tk != 0:
                continue
            if _kw > 1 and (
                _k_per_batch % _kw != 0 or (_k_per_batch // _kw) % _tk != 0
            ):
                continue

            # a16w-mix gemm1 splits TILE_N across (4//k_wave) N-waves, each
            # accumulating (TILE_N//n_waves)//16 MFMA n-tiles. When that is 0
            # (e.g. tile_n=32 with k_wave=1) the epilogue writes nothing ->
            # NaN/garbage output that times fast but is wrong; the tuner would
            # otherwise pick it. Require num_acc_n >= 1 for a16w4.
            if a_dtype_str == "bf16":
                _n_waves = max(1, 4 // _kw)
                # Match the kernel assert; a floor divide lets tile_n=96 through.
                if kparams["tile_n"] % (16 * _n_waves) != 0:
                    continue

            # (kernel_name, kparams, is_fp4, is_fp8)
            # out_dtype encodes fused quant type: "fp4" or "fp8"
            #   a8w4 (a_dtype_str="fp8"): stage2 expects fp8 activations -> out_dtype="fp8"
            #   a4w4 (a_dtype_str="fp4"): stage2 expects fp4 activations -> out_dtype="fp4"
            s1_tile_m = kparams["tile_m"]
            if s1_tile_m != blockM:
                continue
            if a_dtype_str == "fp8":
                fp8_params = {
                    **kparams,
                    "out_dtype": "fp8",
                    "a_scale_one": True,
                    "gate_mode": "interleave",
                }
                nonfused_params = {**kparams, "a_scale_one": True}
                if is_splitk:
                    s1_variants = [(kname + "_fp8", fp8_params, False, True)]
                else:
                    s1_variants = [
                        (kname, nonfused_params, False, False),
                        (kname + "_fp8", fp8_params, False, True),
                    ]
            elif a_dtype_str in ("bf16", "fp16"):
                # a16w4 (bf16/fp16 activation): stage1 emits a bf16
                # intermediate consumed directly by the abf16 stage2 (no
                # inter-stage fp4 quant). Only the bf16-output variant is
                # valid; a fp4-output stage1 would be mispaired with the
                # bf16-input stage2.
                s1_variants = [(kname, kparams, False, False)]
            else:
                fp4_params = {**kparams, "out_dtype": "fp4"}
                if is_splitk:
                    s1_variants = [(kname + "_fp4", fp4_params, True, False)]
                else:
                    s1_variants = [
                        (kname, kparams, False, False),
                        (kname + "_fp4", fp4_params, True, False),
                    ]

            for s1_name, s1_params, is_fp4, is_fp8 in s1_variants:
                s1_compare_fn = None
                if is_fp8 or is_fp4 or a_dtype_str in ("fp8", "fp4"):
                    # Fused stage1 emits packed mx values; the default
                    # comparison reads fp4x2 as raw bytes (two e2m1 codes
                    # each), which rejects >99% code-identical candidates.
                    s1_compare_fn = cosine_diff_compare
                ref_args_extra = (
                    [
                        "a1_qt",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a1_scale",
                        "w1_scale",
                        "sorted_ids",
                        "num_valid_ids",
                        "bias",
                    ],
                    dtype,
                    act_type,
                    q_type,
                    doweight_stage1,
                    topk,
                    blockM,
                )
                if is_fp4:
                    ref_args_extra = ref_args_extra + (True,)
                elif is_fp8:
                    ref_args_extra = ref_args_extra + (False, True)
                s1_ref_func = run_torch_moe_stage1
                s1_ref_args = ref_args_extra
                if a_dtype_str == "bf16":
                    # a16w-mix: kernel returns SORTED output; compare against a
                    # SORTED torch ref (sorting done here in the untimed ref, not
                    # in run_flydsl_stage1_out) so the tuned latency is kernel-only.
                    s1_ref_func = run_a16w_stage1_sorted_ref
                    s1_ref_args = (
                        [
                            "a1_qt",
                            "w1_qt",
                            "w2_qt",
                            "topk_weights",
                            "topk_ids",
                            "w1_scale",
                            "sorted_ids",
                            "sorted_expert_ids",
                            "num_valid_ids",
                        ],
                        dtype,
                        act_type,
                        q_type,
                        doweight_stage1,
                        topk,
                        blockM,
                        inter_dim,
                    )
                    s1_compare_fn = _a16w_sorted_cos
                s1_ref_kwargs = {}
                s1_ref = None

                a1_key = "a1_qt_fp8_cast" if is_fp8 else "a1_qt"
                tasks_flydsl.append(
                    (
                        (info, "stage1", s1_name, blockM),
                        generate_data_2stages,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            doweight_stage1,
                            blockM,
                            1,
                        ),
                        run_flydsl_stage1_out,
                        (
                            [
                                a1_key,
                                "w1_qt_shffle_ck",
                                "sorted_ids",
                                "sorted_expert_ids",
                                "sorted_weights",
                                "num_valid_ids",
                                "w1_scale_aiter",
                                "a1_scale_fp4_sort",
                                "bias",
                            ],
                            dtype,
                            topk,
                            s1_params,
                            blockM,
                            q_dtype_a,
                            q_type,
                            act_type,
                        ),
                        {},
                        s1_ref_func,
                        s1_ref_args,
                        s1_ref_kwargs,
                        s1_ref,
                        0.01,
                        0.01,
                        s1_compare_fn,
                    )
                )

        for kname, kparams in flydsl_s2_kernels.items():
            s2_tile_m = kparams["tile_m"]
            if blockM % s2_tile_m != 0:
                continue
            # Only try matched (tile_m==blockM) and one smaller (blockM/2) to limit candidates
            if s2_tile_m != blockM and s2_tile_m != blockM // 2:
                continue
            # Skip a16w-mix stage2 candidates the port can't run correctly:
            #  - _sbm (tile_m<blockM) re-tiles the SORTED [sorted_size, inter]
            #    stream finer than the moe_sorting padding -> queue fault;
            #  - LDS is tile_m*(tile_k*2 + tile_n*4); over 160 KiB the build fails;
            #  - tile_k must divide inter_dim (K); tile_k=256 on non-256 inter
            #    (e.g. 384) is parsed verbatim by the wrapper -> OOB/wrong out.
            _s2_lds = s2_tile_m * (kparams["tile_k"] * 2 + kparams["tile_n"] * 4)
            if a_dtype_str == "bf16" and (
                s2_tile_m != blockM
                or _s2_lds > 160 * 1024
                or inter_dim % kparams["tile_k"] != 0
            ):
                continue
            # a16w4 (fp4 W) stage2 always atomic-scatters and ignores persist,
            # so its reduce/persist names launch the same kernel as the plain
            # atomic one: tune only that.
            if (
                a_dtype_str == "bf16"
                and b_dtype_str == "fp4"
                and (kparams["mode"] == "reduce" or kparams.get("persist"))
            ):
                continue
            s2_kparams = {**kparams, "sort_block_m": blockM}
            s2_kname = kname if s2_tile_m == blockM else f"{kname}_sbm{blockM}"

            s2_ref_kwargs = {}
            s2_compare_fn = None
            if a_dtype_str == "fp8":
                s2_compare_fn = cosine_diff_compare
                s2_ref_args = (
                    [
                        "ref1_bf16",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a2_scale_none",
                        "w2_scale",
                        "bias",
                    ],
                    dtype,
                    q_type,
                    doweight_stage1,
                )
            else:
                s2_ref_args = (
                    [
                        "a2_qt",
                        "w1_qt",
                        "w2_qt",
                        "topk_weights",
                        "topk_ids",
                        "a2_scale",
                        "w2_scale",
                        "bias",
                    ],
                    dtype,
                    q_type,
                    doweight_stage1,
                )
            s2_ref_func = run_torch_moe_stage2

            tasks_flydsl.append(
                (
                    (info, "stage2", s2_kname, blockM),
                    generate_data_2stages,
                    (
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        act_type,
                        dtype,
                        q_dtype_a,
                        q_dtype_w,
                        q_type,
                        use_g1u1,
                        doweight_stage1,
                        blockM,
                        2,
                    ),
                    run_flydsl_stage2_out,
                    (
                        [
                            ("a2_a16w_sorted" if a_dtype_str == "bf16" else "a2_qt"),
                            "w2_qt_shffle_flydsl",
                            "sorted_ids",
                            "sorted_expert_ids",
                            "sorted_weights",
                            "num_valid_ids",
                            "w2_scale_flydsl",
                            "a2_scale_mxfp4_sort",
                            "moe_buf",
                            "bias",
                        ],
                        dtype,
                        topk,
                        s2_kparams,
                        blockM,
                        q_type,
                        act_type,
                    ),
                    {},
                    s2_ref_func,
                    s2_ref_args,
                    s2_ref_kwargs,
                    (None),
                    0.01,
                    0.01,
                    s2_compare_fn,
                )
            )

    return tasks_flydsl
