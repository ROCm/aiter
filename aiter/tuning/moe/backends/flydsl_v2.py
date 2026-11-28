# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL v2 (MXFP4) candidates of the fused MoE tuner's two-stage path."""

import functools

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    get_flydsl_activation_name,
)
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    get_flydsl_stage1_kernels,
    get_flydsl_stage2_v2_kernels,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import cosine_diff_compare
from aiter.tuning.moe import mxfp4_v2_tune_utils as _v2u
from aiter.tuning.moe.inputs import _ep_route
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    build_v2_inputs as _v2_build_inputs,
)
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    gen as _v2_gen,
)
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    populate_v2_intermediate_from_ref as _v2_populate_stage2,
)
from aiter.tuning.moe.mxfp4_v2_tune_utils import (
    v2_stage1_sorted_ref as _v2_stage1_ref,
)
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import moe_mxfp4_sort


def _v2_gen_ep(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    ep,
    ep_seed=0,
    adtype="fp8",
    b_dtype="fp4",
    activation=ActivationType.Silu,
    situ_beta=DEFAULT_SITUV2_BETA,
    situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
):
    """mxfp4_v2_tune_utils.gen() with EP routing (see _ep_route).

    Builds only the fields the flydslv2 tasks read. ``topk_ids`` holds global
    expert ids for the sort; the torch references use ``local_topk_ids``, where
    remote routes and padding rows are -1.
    """
    torch.manual_seed(0)
    inp = torch.randn((token, model_dim), dtype=dtypes.bf16) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16) / 10
    topk_weights, topk_ids, local_topk_ids, expert_mask, num_local_tokens = _ep_route(
        inp, E, topk, ep, ep_seed
    )

    w1_qt, w1_scale = _v2u.quant_w(w1, b_dtype)
    w2_qt, w2_scale = _v2u.quant_w(w2, b_dtype)
    a1_qt, a1_scale = _v2u.quant_a(inp, adtype)

    a_deq = _v2u._a_deq(a1_qt, a1_scale, token, model_dim, adtype)
    w1_deq = (
        _v2u.dequant_w(w1_qt, w1_scale, b_dtype, E * inter_dim * 2, model_dim)
        .view(E, inter_dim * 2, model_dim)
        .to(dtypes.bf16)
    )
    w2_deq = (
        _v2u.dequant_w(w2_qt, w2_scale, b_dtype, E * model_dim, inter_dim)
        .view(E, model_dim, inter_dim)
        .to(dtypes.bf16)
    )
    ref1 = torch_moe_stage1(
        a_deq,
        w1_deq,
        w2_deq,
        topk_weights,
        local_topk_ids,
        dtype=dtypes.bf16,
        activation=activation,
        quant_type=QuantType.No,
        doweight=False,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    ref2 = torch_moe_stage2(
        ref1,
        w1_deq,
        w2_deq,
        topk_weights,
        local_topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.No,
        doweight=True,
    )
    ref2_quantized = None
    if b_dtype == "fp8":
        a2_ref_q, a2_ref_scale = _v2u.quant_a(
            ref1.contiguous().view(token * topk, inter_dim), adtype
        )
        a2_ref_deq = _v2u._a_deq(
            a2_ref_q, a2_ref_scale, token * topk, inter_dim, adtype
        ).view(token, topk, inter_dim)
        ref2_quantized = torch_moe_stage2(
            a2_ref_deq,
            w1_deq,
            w2_deq,
            topk_weights,
            local_topk_ids,
            dtype=dtypes.bf16,
            quant_type=QuantType.No,
            doweight=True,
        )

    if b_dtype == "fp8":
        w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, True)
        w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, True)
        w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_scale_shuf = fp4_utils.e8m0_shuffle(w2_scale)
    else:
        w1_qt_shuf, w1_scale_shuf = _v2u._baseline_w1_shuffle(
            w1_qt, w1_scale, E, adtype
        )
        w2_qt_shuf = _v2u._mxfp4_shuffle_weight_a16w4(w2_qt, gate_up=False)
        w2_scale_shuf = _v2u._mxfp4_shuffle_scale_a16w4(w2_scale, E, gate_up=False)
    return {
        "ref1": ref1,
        "ref2": ref2,
        "ref2_quantized": ref2_quantized,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "local_topk_ids": local_topk_ids,
        "expert_mask": expert_mask,
        "num_local_tokens": num_local_tokens,
        "a1_qt": a1_qt,
        "a1_scale": a1_scale,
        "inp": inp,
        "base": {
            "w1_qt_shuf": w1_qt_shuf,
            "w1_scale_shuf": w1_scale_shuf,
            "w2_qt_shuf": w2_qt_shuf,
            "w2_scale_shuf": w2_scale_shuf,
        },
        "adtype": adtype,
        "stage2_adtype": adtype,
        "b_dtype": b_dtype,
    }


def _v2_build_inputs_ep(d, model_dim, inter_dim, BM_S1):
    """mxfp4_v2_tune_utils.build_v2_inputs() for _v2_gen_ep data.

    Sorts through the runtime moe_sorting with the global expert count,
    expert_mask and num_local_tokens, as fused_moe does under EP, so the
    sorted buffers (and hence the GEMM grids) match the runtime sizes.
    """
    sti, swt, sei, cumsum, _ = moe_sorting(
        d["topk_ids"],
        d["topk_weights"],
        d["expert_mask"].numel(),
        model_dim,
        dtypes.bf16,
        BM_S1,
        d["expert_mask"],
        d["num_local_tokens"],
    )
    torch.cuda.synchronize()
    max_sorted = sti.shape[0]
    inter_cols = inter_dim if d["stage2_adtype"] == "fp8" else inter_dim // 2
    isq = torch.zeros((max_sorted, inter_cols), dtype=torch.uint8)
    isc_cols = inter_dim // 32
    isr = (
        (((max_sorted * ((2 * inter_dim) // 64) * 4) + isc_cols - 1) // isc_cols + 31)
        // 32
        * 32
    )
    iss = torch.zeros((isr, isc_cols), dtype=torch.uint8)
    return {
        "w2u8": _v2u._u8v(d["base"]["w2_qt_shuf"]),
        "w2sc": _v2u._u8v(d["base"]["w2_scale_shuf"]),
        "sei": sei,
        "cumsum": cumsum,
        "sti": sti,
        "swt": swt,
        "isq": isq,
        "iss": iss,
        "n": int(cumsum[0].item()),
        "max_sorted": max_sorted,
    }


def generate_v2_stage1_data(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    blockM,
    adtype,
    b_dtype,
    act_type,
    ep=0,
    ep_seed=0,
    device="cuda",
):
    # Build the SAME inputs the runtime v2 gemm1 path consumes:
    # aiter/fused_moe.py:_flydsl_stage1_wrapper -> flydsl_moe_stage1(
    #   a=a1_qt, w1=w1_qt_shuf, w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
    #   v2_output_layout=True, ...). Recipe mirrors
    # mxfp4_v2_tune_utils.populate_baseline_v2_intermediate.
    # Generate on the GPU: the CPU fp4 packing of E x 2I x H weights takes
    # tens of minutes per shape at E=385.
    with torch.device(device):
        if ep:
            d = _v2_gen_ep(
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                ep,
                ep_seed,
                adtype=adtype,
                b_dtype=b_dtype,
                activation=act_type,
                situ_beta=DEFAULT_SITUV2_BETA,
                situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
            )
        else:
            d = _v2_gen(
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                blockM,
                adtype=adtype,
                b_dtype=b_dtype,
                activation=act_type,
                situ_beta=DEFAULT_SITUV2_BETA,
                situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
            )
    if ep:
        v = _v2_build_inputs_ep(d, model_dim, inter_dim, blockM)
    else:
        v = _v2_build_inputs(d, token, model_dim, inter_dim, expert, topk, blockM)
    # Precompute the sorted A-scale here so it is NOT timed in the run func.
    a1_scale_sort = moe_mxfp4_sort(
        d["a1_scale"][:token, :].view(token, 1, -1),
        sorted_ids=v["sti"],
        num_valid_ids=v["cumsum"],
        token_num=token,
        block_size=blockM,
    )
    # The a8w4 a_scale_one=True variant uses an unscaled bf16->fp8 cast.
    # The a8w8 path uses the real per-1x32 payload and E8M0 scale instead.
    a1_qt_fp8_cast = (
        d["inp"].to(dtypes.fp8) if adtype == "fp8" and b_dtype == "fp4" else None
    )
    return {
        "a1_qt": d["a1_qt"],
        "a1_qt_fp8_cast": a1_qt_fp8_cast,
        "w1_shuf": d["base"]["w1_qt_shuf"],
        "w1_scale_shuf": d["base"]["w1_scale_shuf"],
        "a1_scale_sort": a1_scale_sort,
        "sti": v["sti"],
        "sei": v["sei"],
        "cumsum": v["cumsum"],
        "isq": v["isq"],
        "n": v["n"],
        "ref1": d["ref1"],
        "topk_ids": d["local_topk_ids"] if ep else d["topk_ids"],
    }


def run_flydsl_v2_stage1_out(
    a1_qt,
    w1_shuf,
    w1_scale_shuf,
    a1_scale_sort,
    sti,
    sei,
    cumsum,
    isq,
    n,
    model_dim,
    inter_dim,
    expert,
    topk,
    blockM,
    adtype,
    act_type,
    kparams,
):
    # Time ONLY the runtime v2 gemm1 kernel: flydsl_moe_stage1(v2_output_layout=True).
    # a1_scale_sort is precomputed in generate_v2_stage1_data (not timed).
    out, _scale = flydsl_moe_stage1(
        a=a1_qt,
        w1=w1_shuf,
        out=isq,
        sorted_token_ids=sti,
        sorted_expert_ids=sei,
        num_valid_ids=cumsum,
        topk=topk,
        tile_m=kparams["tile_m"],
        tile_n=kparams["tile_n"],
        tile_k=kparams["tile_k"],
        a_dtype=kparams["a_dtype"],
        b_dtype=kparams["b_dtype"],
        out_dtype=kparams["out_dtype"],
        act=get_flydsl_activation_name(act_type),
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
        w1_scale=w1_scale_shuf,
        a1_scale=a1_scale_sort,
        sorted_weights=None,
        use_async_copy=True,
        k_batch=1,
        waves_per_eu=kparams.get("waves_per_eu", 3),
        b_nt=kparams.get("b_nt", 2),
        gate_mode=kparams.get("gate_mode", "separated"),
        a_scale_one=kparams.get("a_scale_one", False),
        xcd_swizzle=kparams.get("xcd_swizzle", 0),
        k_wave=kparams.get("k_wave", 1),
        v2_output_layout=True,
    )
    return out.view(torch.uint8).view_as(isq)


def generate_v2_stage2_data(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    blockM,
    adtype,
    b_dtype,
    act_type,
    ep=0,
    ep_seed=0,
    device="cuda",
):
    if ep:
        d = _v2_gen_ep(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            ep,
            ep_seed,
            adtype=adtype,
            b_dtype=b_dtype,
            activation=act_type,
            situ_beta=DEFAULT_SITUV2_BETA,
            situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
        )
        v = _v2_build_inputs_ep(d, model_dim, inter_dim, blockM)
    else:
        d = _v2_gen(
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            blockM,
            adtype=adtype,
            b_dtype=b_dtype,
            activation=act_type,
            situ_beta=DEFAULT_SITUV2_BETA,
            situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
        )
        v = _v2_build_inputs(d, token, model_dim, inter_dim, expert, topk, blockM)
    _v2_populate_stage2(d, v, token, topk, blockM)
    return {
        "isq": v["isq"],
        "iss": v["iss"],
        "w2u8": v["w2u8"],
        "w2sc": v["w2sc"],
        "sei": v["sei"],
        "cumsum": v["cumsum"],
        "sti": v["sti"],
        "swt": v["swt"],
        "n": v["n"],
        "max_sorted": v["max_sorted"],
        "ref2": d["ref2_quantized"] if b_dtype == "fp8" else d["ref2"],
        "expert_mask": d["expert_mask"] if ep else None,
        "topk_ids": d["topk_ids"] if ep else None,
    }


def run_flydsl_v2_stage2_out(
    isq,
    iss,
    w2u8,
    w2sc,
    sei,
    cumsum,
    sti,
    swt,
    n,
    max_sorted,
    ref2,
    expert_mask,
    topk_ids,
    model_dim,
    inter_dim,
    expert,
    topk,
    kparams,
):
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

    is_ep = expert_mask is not None
    token = ref2.shape[0]
    epilog = kparams["epilog"]
    bm_s2 = kparams["tile_m"]
    if epilog == "scatter" and bm_s2 != 128:
        raise AssertionError(f"epilog='scatter' supports only tile_m=128, got {bm_s2}")
    sbm = kparams["sort_block_m"]
    out = torch.empty((token, model_dim), dtype=dtypes.bf16, device=isq.device)
    inter = (
        torch.empty((token, topk, model_dim), dtype=dtypes.bf16, device=isq.device)
        if epilog == "reduce"
        else None
    )
    flat = (
        torch.empty((max_sorted, model_dim), dtype=dtypes.bf16, device=isq.device)
        if epilog == "scatter"
        else None
    )
    gemm2_out = inter if epilog == "reduce" else (flat if epilog == "scatter" else out)
    if epilog != "reduce":
        out.zero_()
    elif is_ep:
        # As _flydsl_v2_stage2_wrapper: EP leaves remote-route slots unwritten.
        inter.zero_()
    mxfp4_moe_gemm2(
        inter_sorted_quant=isq,
        inter_sorted_shuffled_scale=iss,
        w2_u8=w2u8,
        w2_scale_u8=w2sc,
        sorted_expert_ids=sei,
        cumsum_tensor=cumsum,
        sorted_token_ids=sti,
        sorted_weights=swt,
        out=gemm2_out,
        M_logical=token,
        max_sorted=max_sorted,
        NE=expert,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        BM=bm_s2,
        BN=kparams["tile_n"],
        BK=kparams["tile_k"],
        use_nt=kparams["use_nt"],
        a_dtype=kparams["a_dtype"],
        b_dtype=kparams["b_dtype"],
        epilog=epilog,
        SBM=sbm,
        persist=kparams["persist"],
        # The runtime EP call passes no padded count, so the grid is the
        # M_logical * topk upper bound rather than the valid sorted rows.
        n_sorted_padded=None if is_ep else n,
        is_ep=is_ep,
        use_valid_token_count=is_ep,
        g2_scatter_lds_pairs=kparams.get("scatter_lds_pairs", False),
    )
    if epilog == "scatter":
        n_valid = int(n)
        packed = sti[:n_valid]
        token_id = packed & 0x00FFFFFF
        slot = packed >> 24
        ok = (token_id < token) & (slot < topk)
        reverse_sorted = torch.full(
            (token * topk,), -1, dtype=torch.int32, device=sti.device
        )
        reverse_sorted[(token_id[ok] * topk + slot[ok]).long()] = torch.arange(
            n_valid, device=sti.device, dtype=torch.int32
        )[ok]
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=flat,
            reverse_sorted=reverse_sorted,
            sorted_weights=swt,
            out=out,
            NE=expert,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=bm_s2,
        )
    if epilog == "reduce":
        from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

        _run_moe_reduction(
            inter,
            out,
            token,
            topk,
            model_dim,
            expert_mask=expert_mask,
            topk_ids=topk_ids,
        )
    return out


def run_v2_stage2_ref(ref2):
    return ref2


def run_v2_stage1_sorted_ref(ref1, topk_ids, sti, sei, n, token, inter_dim, bm_s1):
    return _v2_stage1_ref(
        ref1,
        topk_ids,
        sti,
        sei,
        n,
        token=token,
        inter_dim=inter_dim,
        bm_s1=bm_s1,
        max_sorted=sti.numel(),
    )


def gen_flydsl_v2_2stages_task(info, blockMs, ep=0, ep_seed=0):
    tasks = []
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
    if dtype != dtypes.bf16:
        return tasks
    if doweight_stage1:
        return tasks
    if q_type != QuantType.per_1x32 or not use_g1u1:
        return tasks
    if q_dtype_a == dtypes.fp4x2 and q_dtype_w == dtypes.fp4x2:
        adtype, bdtype = "fp4", "fp4"
    elif q_dtype_a == dtypes.fp8 and q_dtype_w == dtypes.fp4x2:
        adtype, bdtype = "fp8", "fp4"
    elif q_dtype_a == dtypes.fp8 and q_dtype_w == dtypes.fp8:
        adtype, bdtype = "fp8", "fp8"
    else:
        return tasks

    # v2 always fuses quant; dtype is bf16 out layout at the flydsl kernel level.
    out_dtype_str = "bf16"
    s1_kernels = get_flydsl_stage1_kernels(adtype, bdtype, out_dtype_str)

    from aiter.tuning.moe.mxfp4_v2_tune_utils import (
        v2_stage1_dequant_cosine_err,
    )

    for blockM in blockMs:
        if blockM not in (32, 64, 128):
            continue
        # ---- v2 stage1 tasks: sweep the FULL fused stage1 candidate set ----
        # Mirror gen_flydsl_2stages_task's fused-variant construction so
        # label == timed config == runtime config (kname1 encodes
        # tile_n/waves/b_nt/gate_mode/xcd_swizzle/k_wave). v2 requires
        # non-splitk (k_batch==1: _v2_output_layout = _fuse_any_quant and
        # not _is_splitk, moe_kernels.py:1179) and tile_m == blockM.
        s1_compare = functools.partial(
            v2_stage1_dequant_cosine_err, inter_dim=inter_dim, adtype=adtype
        )
        for kname, kparams in s1_kernels.items():
            if kparams.get("tile_m") != blockM:
                continue
            if kparams.get("k_batch", 1) != 1:
                continue
            # Only the FUSED variant (v2 always fuses quant). Fused name and
            # params exactly as gen_flydsl_2stages_task builds them.
            if adtype == "fp8":
                s1_name = kname + "_fp8"
                s1_params = {
                    **kparams,
                    "out_dtype": "fp8",
                    "a_scale_one": bdtype == "fp4",
                    "gate_mode": "interleave",
                }
            else:
                s1_name = kname + "_fp4"
                s1_params = {**kparams, "out_dtype": "fp4"}
            a1_key = (
                "a1_qt_fp8_cast" if adtype == "fp8" and bdtype == "fp4" else "a1_qt"
            )
            # info tag: (..., blockM, flat_flag=0, v2=1) -- tail[3]=flat (post_process int(tail[3]) default), tail[4]=v2 marker
            tasks.append(
                (
                    (info, "stage1", s1_name, blockM, 0, 1),
                    generate_v2_stage1_data,
                    (
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        blockM,
                        adtype,
                        bdtype,
                        act_type,
                        ep,
                        ep_seed,
                    ),
                    run_flydsl_v2_stage1_out,
                    (
                        [
                            a1_key,
                            "w1_shuf",
                            "w1_scale_shuf",
                            "a1_scale_sort",
                            "sti",
                            "sei",
                            "cumsum",
                            "isq",
                            "n",
                        ],
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        blockM,
                        adtype,
                        act_type,
                        s1_params,
                    ),
                    {},
                    run_v2_stage1_sorted_ref,
                    (
                        ["ref1", "topk_ids", "sti", "sei", "n"],
                        token,
                        inter_dim,
                        blockM,
                    ),
                    {},
                    None,
                    0.1,
                    0.1,
                    s1_compare,
                )
            )
        # ---- v2 stage2 tasks ----
        for kn2, kp in get_flydsl_stage2_v2_kernels(
            adtype,
            bdtype,
            "bf16",
            blockM,
            model_dim=model_dim,
            inter_dim=inter_dim,
        ).items():
            kp = {**kp, "a_dtype": adtype}
            # flat=0, v2=1
            tasks.append(
                (
                    (info, "stage2", kn2, blockM, 0, 1),
                    generate_v2_stage2_data,
                    (
                        token,
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        blockM,
                        adtype,
                        bdtype,
                        act_type,
                        ep,
                        ep_seed,
                    ),
                    run_flydsl_v2_stage2_out,
                    (
                        [
                            "isq",
                            "iss",
                            "w2u8",
                            "w2sc",
                            "sei",
                            "cumsum",
                            "sti",
                            "swt",
                            "n",
                            "max_sorted",
                            "ref2",
                            "expert_mask",
                            "topk_ids",
                        ],
                        model_dim,
                        inter_dim,
                        expert,
                        topk,
                        kp,
                    ),
                    {},
                    run_v2_stage2_ref,
                    (["ref2"],),
                    {},
                    None,
                    0.1,
                    0.1,
                    cosine_diff_compare,
                )
            )
    return tasks
