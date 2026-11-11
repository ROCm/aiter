# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL int4 (a16wi4) candidates of the fused MoE tuner's two-stage path."""

from aiter import QuantType, dtypes
from aiter.ops.flydsl.moe_kernels import (
    get_flydsl_stage1_kernels_int4_bf16,
    get_flydsl_stage2_kernels_int4_bf16,
)
from aiter.test_common import cosine_diff_compare
from aiter.tuning.moe.backends.flydsl import (
    run_a16w_stage1_sorted_ref,
    run_flydsl_stage1_out,
    run_flydsl_stage2_out,
)
from aiter.tuning.moe.compare import _a16w_sorted_cos, run_torch_moe_stage2
from aiter.tuning.moe.inputs import generate_data_2stages


def gen_flydsl_i4_2stages_task(info, blockMs):
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

    if not (q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2):
        return tasks_flydsl

    out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"
    # bf16 -> "bf16" (not "fp16"): a16wi4 IS the a16w-mix bf16-activation path, and
    # a_dtype_str is what selects the SORTED bf16 intermediate ("a2_a16w_sorted")
    # for the stage2 candidates below. Mapping it to "fp16" fed stage2 the packed
    # a2_qt instead, so every stage2 candidate raised and none was ever timed.
    _a_dtype_map = {
        dtypes.fp8: "fp8",
        dtypes.fp4x2: "fp4",
        dtypes.fp16: "fp16",
        dtypes.bf16: "bf16",
    }
    a_dtype_str = _a_dtype_map.get(q_dtype_a, "fp8")

    flydsl_s1_kernels = get_flydsl_stage1_kernels_int4_bf16(out_dtype_str)
    flydsl_s2_kernels = get_flydsl_stage2_kernels_int4_bf16(out_dtype_str)

    for blockM in blockMs:
        if blockM not in [16, 32, 64, 128] or not use_g1u1:
            continue
        for kname, kparams in flydsl_s1_kernels.items():
            # a16wi4 constraint: block_m == kn1.tile_m == kn2.tile_m.
            # Mismatched tile_m breaks correctness, so only consider
            # stage1 kernels whose tile_m equals the current blockM.
            ktm = kparams["tile_m"]
            if ktm != blockM:
                continue
            # Validate split-k compatibility with model_dim
            kb = kparams.get("k_batch", 1)
            if kb > 1:
                if model_dim % kb != 0:
                    continue
                k_per_batch = model_dim // kb
                if k_per_batch % kparams["tile_k"] != 0:
                    continue
                k_tiles = k_per_batch // kparams["tile_k"]
                if k_tiles < 4 or k_tiles % 2 != 0:
                    continue
            # Int4 kernels: no fuse_fp4_quant. a16wi4 is the a16w-mix port, whose
            # stage1 returns a SORTED [max_sorted, inter] intermediate, so it must
            # be scored against the SORTED reference (same as the mxfp4 a16w4 arm
            # in gen_flydsl_2stages_task). Against the unsorted run_torch_moe_stage1
            # every candidate scored ~84% err and the whole shape was rejected
            # ("stage1 and stage2 should be valid together"), so stage1 was never
            # actually tuned.
            ref_args_extra = (
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
            tasks_flydsl.append(
                (
                    (info, "stage1", kname, blockM),
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
                            "a1_qt",
                            "w1_qt_shffle_flydsl",
                            "sorted_ids",
                            "sorted_expert_ids",
                            "sorted_weights",
                            "num_valid_ids",
                            "w1_scale_flydsl",
                            "a1_scale_fp4_sort",
                            "bias",
                        ],
                        dtype,
                        topk,
                        kparams,
                        blockM,
                        q_dtype_a,
                        q_type,
                        act_type,
                    ),
                    {},
                    run_a16w_stage1_sorted_ref,
                    ref_args_extra,
                    {},
                    (None),
                    0.01,
                    0.01,
                    _a16w_sorted_cos,
                )
            )

        for kname, kparams in flydsl_s2_kernels.items():
            # a16wi4 constraint: block_m == kn1.tile_m == kn2.tile_m.
            # Stage2 kernels must match blockM as well; otherwise the
            # tuner can emit a row that fails op_tests/test_moe_2stage.
            s2_tile_m = kparams["tile_m"]
            if s2_tile_m != blockM:
                continue
            if kparams.get("mode", "atomic") != "atomic":
                continue
            if kparams.get("persist", None) is not None:
                continue
            s2_kparams = {**kparams, "sort_block_m": blockM}
            s2_kname = kname

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
