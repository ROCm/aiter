# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Input generators of the fused MoE tuner: weights, routing and per-stage data."""

import os

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_dynamic_mxfp8_quant
from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_i4_quant
from aiter.ops.shuffle import (
    pack_int8_to_packed_int4,
    shuffle_scale_a16w4,
    shuffle_scale_for_int4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.tuning.moe.compare import _token_major_to_sorted_routes, run_torch_moe_stage1
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    v2_stage1_sorted_ref as _v2_stage1_ref,
)
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import moe_mxfp4_sort

TUNE_MOE_EXPERT_BALANCE = (
    os.environ.get("TUNE_MOE_EXPERT_BALANCE", "False").lower() == "true"
)


# Positional order of the dict returned by ``generate_data_1stage``.
# The asm 1-stage tasks select/reorder their kernel inputs by integer index
# (``_data_idx``); work_group looks tensors up by *name* in that dict, so the
# integer indices must be translated to names through this list. Keep it in
# sync with the return dict of ``generate_data_1stage``.
_GEN_DATA_1STAGE_KEYS = [
    "input",  # 0
    "a1_qt",  # 1
    "w1_qt_shffle",  # 2
    "w2_qt_shffle",  # 3
    "sorted_ids",  # 4
    "sorted_weights",  # 5
    "sorted_expert_ids",  # 6
    "num_valid_ids",  # 7
    "moe_buf",  # 8
    "a1_scale",  # 9
    "w1_scale",  # 10
    "w2_scale",  # 11
    "w1_qt",  # 12
    "w2_qt",  # 13
    "topk_weights",  # 14
    "topk_ids",  # 15
    "fc1_smooth_scale",  # 16
    "fc2_smooth_scale",  # 17
    "a1_scale_t",  # 18
    "w1_scale_shuffled",  # 19
    "w2_scale_shuffled",  # 20
]


def weight_quant(
    weight,
    qType,
    quant_dtype,
):
    E, dim1, dim2 = weight.shape
    if qType == aiter.QuantType.per_Tensor and quant_dtype != torch.int4:
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight.view(E, -1), quant_dtype=quant_dtype
        )
    elif qType == QuantType.per_1x128:
        weight_qt = (
            weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, -1, 128 * 128)
        )
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight_qt, quant_dtype=quant_dtype
        )
        weight_qt = weight_qt.view(E, -1)
        weight_qt = (
            weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, dim1, dim2)
        )
    elif (
        qType == aiter.QuantType.per_Tensor and quant_dtype == torch.int4
    ):  # int4 w quant
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight.view(E, -1), quant_dtype=dtypes.i8, dtypeMax=7
        )
    elif (
        qType == aiter.QuantType.per_Token and quant_dtype == torch.int4
    ):  # int4 w quant
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight, quant_dtype=dtypes.i8, dtypeMax=7
        )
    elif qType == QuantType.per_1x32 and quant_dtype == dtypes.i4x2:
        weight_qt, weight_scale = per_1x32_i4_quant(weight)
    elif qType == QuantType.per_1x32 and quant_dtype == dtypes.fp8:  # mxfp8
        weight_qt, weight_scale = per_1x32_f8_scale_f8_quant(
            weight, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        torch_quant = aiter.get_torch_quant(qType)
        weight_qt, weight_scale = torch_quant(weight, quant_dtype=quant_dtype)
    return weight_qt, weight_scale


def generate_data(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    blockM,
    device="cuda",
):
    torch.manual_seed(0)
    input = torch.randn((token, model_dim), dtype=dtype, device=device) / 10
    if use_g1u1:
        w1 = (
            torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype, device=device)
            / 10
        )
    else:
        w1 = (
            torch.randn((expert, inter_dim, model_dim), dtype=dtype, device=device) / 10
        )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype, device=device)
    w1_qt, w1_scale = weight_quant(w1, q_type, quant_dtype=q_dtype_w)
    w2_qt, w2_scale = weight_quant(w2, q_type, quant_dtype=q_dtype_w)
    if q_dtype_w is not dtypes.fp4x2:
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    else:
        w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    if TUNE_MOE_EXPERT_BALANCE:
        score = torch.zeros((token, expert), dtype=dtype, device=device)
        start_col = 0
        end_col = topk
        for token_id in range(token):
            score[token_id, start_col:end_col] = 1.0
            start_col = end_col % expert
            end_col = start_col + topk
    else:
        routing_seed = os.environ.get("TUNE_MOE_ROUTING_SEED")
        if routing_seed is not None:
            torch.manual_seed(int(routing_seed))
        score = torch.randn((token, expert), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    if q_type == QuantType.per_1x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=q_dtype_a
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif (
        q_type == aiter.QuantType.per_1x32
        and (q_dtype_a in [dtypes.bf16, dtypes.fp16, dtypes.fp8])
        and q_dtype_w == dtypes.fp4x2
    ) or (
        q_type == aiter.QuantType.per_1x32
        and q_dtype_a == dtypes.fp8
        and q_dtype_w == dtypes.fp8
    ):  # a16w4 / a8w4 / mxfp8 (runtime fuses the fp8 a-quant)
        a1_qt = input.to(dtype)
        a1_scale = None
    elif q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:  # a16wi4
        a1_qt = input.to(dtypes.bf16)
        a1_scale = None
    else:
        torch_quant = aiter.get_torch_quant(q_type)
        a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)
    del w1, w2, score
    if q_dtype_w is not dtypes.fp4x2:
        w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
        w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))
    else:
        w1_qt_shffle = w1_qt
        w2_qt_shffle = w2_qt

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, expert, model_dim, dtype, blockM
    )
    needed = sorted_expert_ids.shape[0] * blockM
    if sorted_ids.shape[0] < needed:
        pad = torch.full(
            (needed - sorted_ids.shape[0],),
            token,
            dtype=sorted_ids.dtype,
            device=sorted_ids.device,
        )
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat(
            [
                sorted_weights,
                torch.zeros(
                    pad.shape[0],
                    dtype=sorted_weights.dtype,
                    device=sorted_weights.device,
                ),
            ]
        )
    return {
        "input": input,
        "a1_qt": a1_qt,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "w1_qt_shffle": w1_qt_shffle,
        "w2_qt_shffle": w2_qt_shffle,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "moe_buf": moe_buf,
        "a1_scale": a1_scale,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }


def generate_asm_stage1(
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
    device="cuda",
):
    _data = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    a1_qt = _data["a1_qt"]
    w1_qt = _data["w1_qt"]
    w2_qt = _data["w2_qt"]
    w1_qt_shffle = _data["w1_qt_shffle"]
    w2_qt_shffle = _data["w2_qt_shffle"]
    sorted_ids = _data["sorted_ids"]
    sorted_weights = _data["sorted_weights"]
    sorted_expert_ids = _data["sorted_expert_ids"]
    num_valid_ids = _data["num_valid_ids"]
    topk_ids = _data["topk_ids"]
    topk_weights = _data["topk_weights"]
    a1_scale = _data["a1_scale"]
    w1_scale = _data["w1_scale"]
    if q_type == QuantType.per_1x128:
        ratio = a1_scale.element_size() // a1_qt.element_size()
        out1 = torch.zeros(
            (token + (token * ratio + 127) // 128, topk, inter_dim),
            dtype=a1_qt.dtype,
            device=device,
        )
    else:
        out1 = torch.empty(
            (token, topk, inter_dim),
            dtype=dtype,
            device=device,
        )
    a1_scale_t = a1_scale
    if q_type == QuantType.per_1x128:
        a1_scale_t = a1_scale.t().contiguous()
    return {
        "a1_qt": a1_qt,
        "w1_qt_shffle": w1_qt_shffle,
        "w2_qt_shffle": w2_qt_shffle,
        "sorted_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "sorted_weights": sorted_weights,
        "num_valid_ids": num_valid_ids,
        "out1": out1,
        "a1_scale_t": a1_scale_t,
        "w1_scale": w1_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "a1_scale": a1_scale,
    }


def generate_data_2stages(
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
    stage=1,
    use_real_mxfp8=False,
    device="cuda",
):
    _data = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    input = _data["input"]
    a1_qt = _data["a1_qt"]
    w1_qt = _data["w1_qt"]
    w2_qt = _data["w2_qt"]
    w1_qt_shffle = _data["w1_qt_shffle"]
    w2_qt_shffle = _data["w2_qt_shffle"]
    sorted_ids = _data["sorted_ids"]
    sorted_weights = _data["sorted_weights"]
    sorted_expert_ids = _data["sorted_expert_ids"]
    num_valid_ids = _data["num_valid_ids"]
    topk_ids = _data["topk_ids"]
    topk_weights = _data["topk_weights"]
    moe_buf = _data["moe_buf"]
    a1_scale = _data["a1_scale"]
    w1_scale = _data["w1_scale"]
    w2_scale = _data["w2_scale"]
    # Pre-bind so branches that skip shuffle_scale_* still reach `is None` below.
    w1_scale_aiter = None
    w2_scale_aiter = None
    if q_dtype_w == torch.int4 and q_type != QuantType.per_1x32:
        w1_qt_shffle_ck = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w1_qt, (16, 16), use_int4=True))
        )
        w2_qt_shffle_ck = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w2_qt, (16, 16), use_int4=True))
        )
    elif q_dtype_w == dtypes.fp4x2 and q_dtype_a == dtypes.fp4x2:
        w1_qt_shffle_ck = shuffle_weight(w1_qt, (16, 16))
        w2_qt_shffle_ck = shuffle_weight(w2_qt, (16, 16))
    elif q_dtype_w == dtypes.fp4x2 and q_dtype_a == dtypes.fp8:
        # a8w4 per_1x32 stage1 just support tune a1_cast now.
        w1_qt_shffle_ck = shuffle_weight_a16w4(w1_qt, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, expert, True)
        w2_qt_shffle_ck = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_scale_aiter = shuffle_scale_a16w4(w2_scale, expert, False)
    elif (
        q_dtype_w == dtypes.fp8
        and q_dtype_a == dtypes.fp8
        and q_type == QuantType.per_1x32
    ):  # mxfp8 (a8w8); per_1x128/per_Token fp8 must use the standard
        # shuffle_weight((16,16)) + e8m0_shuffle scale path (see else below),
        # matching op_tests/test_moe_2stage.py is_mxfp8 routing.
        w1_qt_shffle_ck = shuffle_weight_a16w4(w1_qt, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, expert, True)
        w2_qt_shffle_ck = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif q_dtype_w == dtypes.fp4x2 and q_dtype_a in (dtypes.bf16, dtypes.fp16):
        # a16w4 mxfp4 (bf16/fp16 activation, SiTUv2): standard GGUU (separated
        # gate/up) W1 layout, matching main (moe_kernels a16w4 dispatch uses
        # w_layout="standard"). w2 has no gate/up (gate_up=False), scale via
        # plain e8m0_shuffle.
        w1_qt_shffle_ck = shuffle_weight_a16w4(w1_qt, 16, False)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, expert, False)
        w2_qt_shffle_ck = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    else:
        w1_qt_shffle_ck = w1_qt_shffle
        w2_qt_shffle_ck = w2_qt_shffle
    if q_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
        # a16wi4 consumes the OLD FlyDSL int4 kernel's preshuffle:
        # pack_int8_to_packed_int4(shuffle_weight(w.i8, (16,16))) kpack=8 +
        # shuffle_scale_for_int4 (E,G//2,N,2). Byte-identical caller contract to the
        # replaced kernel -- MUST match op_tests/test_moe_2stage.py / production.
        E1, N1, K1 = w1_qt.shape
        E2, N2, K2 = w2_qt.shape
        w1_qt_shffle_flydsl = (
            pack_int8_to_packed_int4(shuffle_weight(w1_qt.view(dtypes.i8), (16, 16)))
            .view(E1, N1, K1 // 2)
            .view(dtypes.i4x2)
        )
        w2_qt_shffle_flydsl = (
            pack_int8_to_packed_int4(shuffle_weight(w2_qt.view(dtypes.i8), (16, 16)))
            .view(E2, N2, K2 // 2)
            .view(dtypes.i4x2)
        )
        w1_scale_flydsl = (
            shuffle_scale_for_int4(w1_scale, group_size=32).view(-1).contiguous()
        )
        w2_scale_flydsl = (
            shuffle_scale_for_int4(w2_scale, group_size=32).view(-1).contiguous()
        )
        w1_qt_shffle_ck = w1_qt_shffle
        w2_qt_shffle_ck = w2_qt_shffle
        w1_scale_aiter = w1_scale
        w2_scale_aiter = w2_scale
    else:
        if w1_scale_aiter is None:
            w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
            w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

        w1_qt_shffle_flydsl = w1_qt_shffle_ck
        w2_qt_shffle_flydsl = w2_qt_shffle_ck
        w1_scale_flydsl = w1_scale_aiter
        w2_scale_flydsl = w2_scale_aiter

    if stage == 1:
        if not doweight_stage1:
            sorted_weights = None
        if (
            q_type == QuantType.per_1x32
            and q_dtype_w != dtypes.i4x2
            and q_dtype_a == dtypes.fp4x2
        ):
            a1_scale_fp4_sort = moe_mxfp4_sort(
                a1_scale,  # a1_scale[: token * topk, :].view(token, topk, -1),
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token,
                block_size=max(32, blockM),
            )
        else:
            a1_scale_fp4_sort = a1_scale

        # For the _fp8 FlyDSL variant (a_scale_one=True): cast bf16 input to fp8.
        a1_qt_fp8_cast = input.to(dtypes.fp8)
        a1_scale_e8m0_one_sort = None
        if (
            q_type == QuantType.per_1x32
            and q_dtype_a == dtypes.fp8
            and q_dtype_w == dtypes.fp4x2
        ):
            scale_cols = ((model_dim // 32) + 7) // 8 * 8
            a1_scale_e8m0_one_sort = torch.ones(
                (sorted_ids.shape[0], scale_cols),
                dtype=dtypes.fp8_e8m0,
                device=device,
            )
            if use_real_mxfp8:
                a1_fp8_bytes, a1_scale_mxfp8 = torch_dynamic_mxfp8_quant(input)
                a1_qt_fp8_cast = a1_fp8_bytes.view(dtypes.fp8)
                a1_scale_e8m0_one_sort = moe_mxfp4_sort(
                    a1_scale_mxfp8,
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=max(32, blockM),
                )

        return {
            "a1_qt": a1_qt,
            "w1_qt_shffle_ck": w1_qt_shffle_ck,
            "w2_qt_shffle_ck": w2_qt_shffle_ck,
            "a1_scale": a1_scale,
            "w1_scale": w1_scale,
            "sorted_ids": sorted_ids,
            "sorted_expert_ids": sorted_expert_ids,
            "sorted_weights": sorted_weights,
            "num_valid_ids": num_valid_ids,
            "moe_buf": moe_buf,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "a1_scale_fp4_sort": a1_scale_fp4_sort,
            "w1_scale_aiter": w1_scale_aiter,
            "w1_qt_shffle_flydsl": w1_qt_shffle_flydsl,
            "w2_qt_shffle_flydsl": w2_qt_shffle_flydsl,
            "w1_scale_flydsl": w1_scale_flydsl,
            "w2_scale_flydsl": w2_scale_flydsl,
            "a1_qt_fp8_cast": a1_qt_fp8_cast,
            "a1_scale_e8m0_one_sort": a1_scale_e8m0_one_sort,
            "a1_scale_none": None,
            "bias": (
                torch.clamp(
                    torch.randn((expert, inter_dim * 2), dtype=dtype, device=device),
                    -1.0,
                    1.0,
                ).to(torch.float32)
                if (
                    act_type == ActivationType.Swiglu
                    and q_type == QuantType.per_1x32
                    and q_dtype_a == dtypes.fp8
                    and dtype in [dtypes.bf16, dtypes.fp16]
                )
                else None
            ),
        }
    elif stage == 2:
        # a8w4: a1_scale is dummy non-None -> torch_moe_stage1's per_1x32
        # branch would call mxfp4_to_f32(bf16), pass None to take a16w4 path.
        ref_a1_scale = (
            None
            if (q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp8)
            else a1_scale
        )
        ref1 = run_torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            a1_scale=ref_a1_scale,
            w1_scale=w1_scale,
            dtype=dtype,
            activation=act_type,
            quant_type=q_type,
            doweight_stage1=doweight_stage1,
            topk=topk,
        )
        # ref1 is always bf16
        ref1_bf16 = ref1
        a2_a16w_sorted = None  # a16w-mix stage2 pre-sorted input (set below)

        if q_type == QuantType.per_1x128:
            ref1, ref_scale = aiter.pertoken_quant(
                ref1.view(ref1.shape[0], -1, 128), quant_dtype=q_dtype_a
            )
            ref1 = ref1.view(ref1.shape[0], topk, -1)
            ref_scale = ref_scale.view(token, -1)
            a2_qt = ref1
            a2_scale = ref_scale
            a2_scale_mxfp4_sort = a2_scale
        elif q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp4x2:
            torch_quant = aiter.get_torch_quant(q_type)
            a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
            a2_scale_mxfp4_sort = moe_mxfp4_sort(
                a2_scale[: token * topk, :].view(token, topk, -1),
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid_ids,
                token_num=token,
                block_size=blockM,
            )
        elif (
            q_type == QuantType.per_1x32
            and q_dtype_a in (dtypes.bf16, dtypes.fp16)
            and q_dtype_w in (dtypes.fp4x2, dtypes.i4x2)
        ) or (
            q_type == QuantType.No
            and q_dtype_a == dtypes.bf16
            and q_dtype_w == dtypes.bf16
        ):
            # a16w-mix (mxfp4 a16w4 / int4 a16wi4 / bf16 a16w16): the abf16
            # stage2 consumes the bf16 intermediate directly -- no inter-stage
            # activation quant
            # (a2_scale=None). The kernel wants it in SORTED [max_sorted, inter]
            # layout; build that once here (untimed) so run_flydsl_stage2_out stays
            # kernel-only. Without this a16wi4 falls through to the generic
            # per_1x32 torch_quant below, which hands stage2 a packed
            # [token, topk, inter/32] tensor -- every candidate then dies on
            # "requires D_INTER (K) % 64 == 0, got D_INTER=8" and stage2 is
            # effectively untuned.
            a2_qt = ref1
            a2_scale = None
            a2_scale_mxfp4_sort = None
            a2_a16w_sorted = _v2_stage1_ref(
                ref1.view(token, topk, inter_dim),
                topk_ids,
                sorted_ids,
                sorted_expert_ids,
                int(num_valid_ids.reshape(-1)[0].item()),
                token=token,
                inter_dim=inter_dim,
                bm_s1=blockM,
                max_sorted=sorted_ids.numel(),
            )
        elif q_type == QuantType.per_1x32 and q_dtype_a == dtypes.fp8:
            # FlyDSL stage2 receives fp8 input
            a2_qt = ref1.to(dtypes.fp8)
            M = sorted_ids.shape[0]
            N = a2_qt.shape[-1]
            scaleN_pad = ((N // 32) + 7) // 8 * 8
            a2_scale = torch.ones(
                [token * topk, scaleN_pad],
                dtype=dtypes.fp8_e8m0,
                device=a2_qt.device,
            )
            a2_scale_mxfp4_sort = torch.ones(
                [M, scaleN_pad], dtype=dtypes.fp8_e8m0, device=a2_qt.device
            )

            if use_real_mxfp8:
                a2_opus_bytes, a2_opus_scale = torch_dynamic_mxfp8_quant(ref1)
                a2_qt = a2_opus_bytes.view(dtypes.fp8)
                a2_scale_mxfp4_sort = moe_mxfp4_sort(
                    a2_opus_scale[: token * topk, :].view(token, topk, -1),
                    sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token,
                    block_size=max(32, blockM),
                )
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
            a2_scale_mxfp4_sort = a2_scale
        a2_qt = a2_qt.view(token, topk, -1)
        if doweight_stage1:
            sorted_weights = None

        return {
            "a2_qt": a2_qt,
            "w1_qt_shffle_ck": w1_qt_shffle_ck,
            "w2_qt_shffle_ck": w2_qt_shffle_ck,
            "a2_scale": a2_scale,
            "w2_scale": w2_scale,
            "sorted_ids": sorted_ids,
            "sorted_expert_ids": sorted_expert_ids,
            "sorted_weights": sorted_weights,
            "num_valid_ids": num_valid_ids,
            "moe_buf": moe_buf,
            "w1_qt": w1_qt,
            "w2_qt": w2_qt,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "a2_scale_mxfp4_sort": a2_scale_mxfp4_sort,
            "a2_a16w_sorted": a2_a16w_sorted,
            "w2_scale_aiter": w2_scale_aiter,
            "w1_qt_shffle_flydsl": w1_qt_shffle_flydsl,
            "w2_qt_shffle_flydsl": w2_qt_shffle_flydsl,
            "w1_scale_flydsl": w1_scale_flydsl,
            "w2_scale_flydsl": w2_scale_flydsl,
            "ref1_bf16": ref1_bf16,
            "a2_scale_none": None,
            "bias": (
                torch.clamp(
                    torch.randn((expert, model_dim), dtype=dtype, device=device),
                    -1.0,
                    1.0,
                ).to(torch.float32)
                if (
                    act_type == ActivationType.Swiglu
                    and q_type == QuantType.per_1x32
                    and q_dtype_a == dtypes.fp8
                    and dtype in [dtypes.bf16, dtypes.fp16]
                )
                else None
            ),
        }


def generate_opus_sorted_stage2_data(*args, **kwargs):
    data = generate_data_2stages(*args, **kwargs)
    data["a2_qt_sorted"] = _token_major_to_sorted_routes(
        data["a2_qt"],
        data["sorted_ids"],
        data["num_valid_ids"],
        data["moe_buf"].shape[0],
    )
    return data


def generate_data_1stage(
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
    blockM=32,
    device="cuda",
):
    _data = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    input = _data["input"]
    a1_qt = _data["a1_qt"]
    w1_qt = _data["w1_qt"]
    w2_qt = _data["w2_qt"]
    w1_qt_shffle = _data["w1_qt_shffle"]
    w2_qt_shffle = _data["w2_qt_shffle"]
    sorted_ids = _data["sorted_ids"]
    sorted_weights = _data["sorted_weights"]
    sorted_expert_ids = _data["sorted_expert_ids"]
    num_valid_ids = _data["num_valid_ids"]
    topk_ids = _data["topk_ids"]
    topk_weights = _data["topk_weights"]
    moe_buf = _data["moe_buf"]
    a1_scale = _data["a1_scale"]
    w1_scale = _data["w1_scale"]
    w2_scale = _data["w2_scale"]
    a1_scale_t = a1_scale
    if q_type == QuantType.per_1x128:
        a1_scale_t = a1_scale.t().contiguous()
    ##smooth scale
    # [expert, 1, model_dim]
    fc1_smooth_scale = torch.randn(
        (expert, 1, model_dim), dtype=dtypes.fp32, device=device
    )
    # [expert, 1, inter_dim]
    fc2_smooth_scale = torch.randn(
        (expert, 1, inter_dim), dtype=dtypes.fp32, device=device
    )
    fc1_smooth_scale = None
    fc2_smooth_scale = None
    w1_scale_shuffled = None
    w2_scale_shuffled = None
    if q_type == QuantType.per_1x32:
        a1_scale = moe_mxfp4_sort(
            a1_scale,
            sorted_ids,
            num_valid_ids,
            token,
            blockM,
        )
        w1_scale = w1_scale.view(expert, -1)
        w2_scale = w2_scale.view(expert, -1)
        if q_dtype_a == q_dtype_w == dtypes.fp4x2:
            w1_qt_shffle = shuffle_weight(w1_qt, layout=(16, 16))
            w2_qt_shffle = shuffle_weight(w2_qt, layout=(16, 16))
            w1_scale_shuffled = fp4_utils.e8m0_shuffle(_data["w1_scale"])
            w2_scale_shuffled = fp4_utils.e8m0_shuffle(_data["w2_scale"])

    return {
        "input": input,
        "a1_qt": a1_qt,
        "w1_qt_shffle": w1_qt_shffle,
        "w2_qt_shffle": w2_qt_shffle,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "moe_buf": moe_buf,
        "a1_scale": a1_scale,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w1_qt": w1_qt,
        "w2_qt": w2_qt,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "fc1_smooth_scale": fc1_smooth_scale,
        "fc2_smooth_scale": fc2_smooth_scale,
        "a1_scale_t": a1_scale_t,
        "w1_scale_shuffled": w1_scale_shuffled,
        "w2_scale_shuffled": w2_scale_shuffled,
    }
