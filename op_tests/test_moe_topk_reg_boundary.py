# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Boundary coverage for the register-resident grouped top-k path.

The GLM-5.2 production route is biased sigmoid, E=256, k=8, G=1,
renormalization enabled, and routed_scaling_factor=2.5.
"""

import itertools
import zlib

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ("gfx942", "gfx950")
REG_EXPERTS = (128, 192, 256, 384, 512, 896, 1024, 2048)
REG_EXPERT_SET = frozenset(REG_EXPERTS)
LEGACY_LANE_PRIORITY = (
    63,
    62,
    61,
    60,
    56,
    57,
    58,
    59,
    48,
    49,
    50,
    51,
    55,
    54,
    53,
    52,
    47,
    46,
    45,
    44,
    40,
    41,
    42,
    43,
    32,
    33,
    34,
    35,
    39,
    38,
    37,
    36,
    31,
    30,
    29,
    28,
    24,
    25,
    26,
    27,
    16,
    17,
    18,
    19,
    23,
    22,
    21,
    20,
    15,
    14,
    13,
    12,
    8,
    9,
    10,
    11,
    0,
    1,
    2,
    3,
    7,
    6,
    5,
    4,
)


def _make_view(values: torch.Tensor, layout: str) -> torch.Tensor:
    token, expert = values.shape
    if layout == "contiguous":
        return values.clone()
    if layout == "odd_stride":
        backing = torch.empty((token, expert + 1), dtype=values.dtype)
        view = backing[:, :expert]
    elif layout == "even_stride":
        backing = torch.empty((token, expert + 2), dtype=values.dtype)
        view = backing[:, :expert]
    elif layout == "even_stride_odd_offset":
        backing = torch.empty((token, expert + 2), dtype=values.dtype)
        view = backing[:, 1 : expert + 1]
    else:
        raise ValueError(f"unknown layout: {layout}")
    view.copy_(values)
    return view


def _make_bias(values: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        return values.clone()
    if layout == "odd_offset":
        backing = torch.empty(values.numel() + 1, dtype=values.dtype)
        view = backing[1:]
        view.copy_(values)
        return view
    raise ValueError(f"unknown bias layout: {layout}")


def _owned_experts(lane: int, experts_per_lane: int) -> list[int]:
    nvec = experts_per_lane // 2
    ids = [(lane + j * 64) * 2 + i for j, i in itertools.product(range(nvec), range(2))]
    if experts_per_lane % 2:
        ids.append(2 * nvec * 64 + lane)
    return ids


def _legacy_plateau_ids(expert: int, topk: int, last_n: int | None = None) -> list[int]:
    start = 0 if last_n is None else expert - last_n
    allowed = set(range(start, expert))
    ordered = []
    for lane in LEGACY_LANE_PRIORITY:
        for vec4 in range(lane, expert // 4, 64):
            ordered.extend(
                expert_id
                for expert_id in range(vec4 * 4, vec4 * 4 + 4)
                if expert_id in allowed
            )
    return ordered[:topk]


def _make_values(
    case: str,
    token: int,
    expert: int,
    dtype: torch.dtype,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(zlib.crc32(case.encode()))
    logits = torch.randn((token, expert), generator=generator, dtype=dtypes.fp32)
    bias = torch.randn((expert,), generator=generator, dtype=dtypes.fp32) * 0.1

    if pattern == "all_equal":
        logits.zero_()
        bias.zero_()
    elif pattern == "cutoff_tie":
        logits.fill_(-4.0)
        logits[:, :12] = 4.0
        bias.zero_()
    elif pattern == "sixteen_way_tie":
        logits.fill_(-4.0)
        logits[:, -16:] = 0.0
        bias.zero_()
    elif pattern == "saturated":
        logits[:, 0::2] = 80.0
        logits[:, 1::2] = -80.0
        bias.zero_()
    elif pattern == "all_nan":
        logits.fill_(torch.nan)
        bias.zero_()
    elif pattern == "all_neg_inf_selection":
        logits.zero_()
        bias.fill_(-torch.inf)
    elif pattern in ("pivot64", "pivot65"):
        assert expert == 2048
        logits.fill_(-10.0)
        counts = (16, 16, 16, 16) if pattern == "pivot64" else (17, 16, 16, 16)
        for lane, (count, value) in enumerate(zip(counts, (4.0, 3.0, 2.0, 1.0))):
            logits[:, _owned_experts(lane, expert // 64)[:count]] = value
        bias.zero_()
    elif pattern != "random":
        raise ValueError(f"unknown pattern: {pattern}")

    return logits.to(dtype), bias.to(dtype)


def _scores(
    logits: torch.Tensor, bias: torch.Tensor | None, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_f32 = logits.to(dtypes.fp32)
    if mode == "softmax":
        output_scores = torch.softmax(logits_f32, dim=-1)
    elif mode in ("sigmoid", "biased"):
        output_scores = torch.sigmoid(logits_f32)
    else:
        raise ValueError(f"unknown mode: {mode}")
    choice_scores = output_scores if mode != "biased" else output_scores + bias.float()
    return output_scores, choice_scores


def _eligible_scores(
    choice_scores: torch.Tensor,
    group: int,
    topk_group: int,
    mode: str,
) -> torch.Tensor:
    if topk_group == group:
        return choice_scores
    token, expert = choice_scores.shape
    grouped = choice_scores.view(token, group, expert // group)
    if mode == "biased":
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    else:
        group_scores = grouped.max(dim=-1).values
    group_ids = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_ids, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(token, group, expert // group)
        .reshape(token, expert)
    )
    return choice_scores.masked_fill(~expert_mask, -torch.inf)


def _expected_route(expert: int, topk: int, group: int, topk_group: int) -> str:
    if (
        get_gfx() in SUPPORTED_GFX
        and expert in REG_EXPERT_SET
        and 4 <= topk <= 32
        and topk_group == group
    ):
        return "register"
    return "lds_fallback"


def _run_kernel(
    logits: torch.Tensor,
    bias: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    mode: str,
    group: int,
    topk_group: int,
    need_renorm: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "biased":
        aiter.biased_grouped_topk_hip(
            logits,
            bias,
            weights,
            ids,
            group,
            topk_group,
            need_renorm,
            scale,
        )
    else:
        aiter.grouped_topk(
            logits,
            weights,
            ids,
            group,
            topk_group,
            need_renorm,
            mode == "softmax",
            scale,
        )
    return weights, ids


@benchmark()
def test_moe_topk_reg_boundary(
    case: str,
    mode: str,
    token: int,
    expert: int,
    topk: int,
    group: int,
    topk_group: int,
    dtype: torch.dtype,
    layout: str,
    bias_layout: str,
    output_pad: int,
    need_renorm: bool,
    scale: float,
    pattern: str,
):
    logits_values, bias_values = _make_values(case, token, expert, dtype, pattern)
    logits = _make_view(logits_values, layout)
    bias = _make_bias(bias_values, bias_layout)

    output_stride = topk + output_pad
    weight_storage = torch.full((token, output_stride), torch.nan, dtype=dtypes.fp32)
    id_sentinel = -777777
    id_storage = torch.full((token, output_stride), id_sentinel, dtype=dtypes.i32)
    weights = weight_storage[:, :topk]
    ids = id_storage[:, :topk]

    candidates = {
        "aiter": lambda: _run_kernel(
            logits,
            bias,
            weights,
            ids,
            mode,
            group,
            topk_group,
            need_renorm,
            scale,
        )
    }

    # One score transform/comparison per input is a conservative operation count.
    ops = token * expert
    nbytes = (
        token * expert * logits.element_size()
        + (expert * bias.element_size() if mode == "biased" else 0)
        + token * topk * (weights.element_size() + ids.element_size())
    )
    ret = {
        "gfx": get_gfx(),
        "expected route": _expected_route(expert, topk, group, topk_group),
        "input stride": logits.stride(0),
        "input offset": logits.storage_offset(),
        "bias offset": bias.storage_offset(),
    }

    output_scores, choice_scores = _scores(
        logits, bias if mode == "biased" else None, mode
    )
    eligible_scores = _eligible_scores(choice_scores, group, topk_group, mode)
    kth = eligible_scores.topk(topk, dim=-1, sorted=False).values.min(dim=-1).values

    legacy_weights = None
    legacy_ids = None
    if (
        mode == "biased"
        and expert == 256
        and group == 1
        and topk_group == 1
        and 4 <= topk <= 32
    ):
        legacy_logits = torch.full((token, expert + 4), -torch.inf, dtype=dtype)
        legacy_logits[:, :expert].copy_(logits)
        legacy_bias = torch.full((expert + 4,), -torch.inf, dtype=dtype)
        legacy_bias[:expert].copy_(bias)
        legacy_weight_storage = torch.empty((token, output_stride), dtype=dtypes.fp32)
        legacy_id_storage = torch.empty((token, output_stride), dtype=dtypes.i32)
        legacy_weights = legacy_weight_storage[:, :topk]
        legacy_ids = legacy_id_storage[:, :topk]
        _run_kernel(
            legacy_logits,
            legacy_bias,
            legacy_weights,
            legacy_ids,
            mode,
            group,
            topk_group,
            need_renorm,
            scale,
        )

    for name, fn in candidates.items():
        (actual_weights, actual_ids), us = run_perftest(fn, num_iters=2, num_warmup=1)
        actual_ids_i64 = actual_ids.to(torch.int64)
        ids_in_range = bool(
            ((actual_ids_i64 >= 0) & (actual_ids_i64 < expert)).all().item()
        )
        sorted_ids = actual_ids_i64.sort(dim=-1).values
        ids_unique = bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all().item())
        finite = bool(torch.isfinite(actual_weights).all().item())
        weight_guard = bool(torch.isnan(weight_storage[:, topk:]).all().item())
        id_guard = bool((id_storage[:, topk:] == id_sentinel).all().item())
        legacy_tie_compatible = True
        if expert in REG_EXPERT_SET and pattern in ("all_equal", "sixteen_way_tie"):
            # Strict-`>` warp argmax in the replaced float4 LDS path resolves
            # plateaus by this lane/local order. A different equal-score set
            # still changes which expert functions run.
            last_n = 16 if pattern == "sixteen_way_tie" else None
            expected_ids = torch.tensor(
                _legacy_plateau_ids(expert, topk, last_n),
                dtype=dtypes.i32,
            ).expand(token, -1)
            legacy_tie_compatible = bool(torch.equal(actual_ids, expected_ids))
        elif pattern in ("all_nan", "all_neg_inf_selection"):
            expected_ids = torch.arange(topk, dtype=dtypes.i32).expand(token, -1)
            legacy_tie_compatible = bool(torch.equal(actual_ids, expected_ids))

        weight_err = 1.0
        max_selection_gap = torch.inf
        order_inversion_ratio = 1.0
        degenerate_pattern = pattern in ("all_nan", "all_neg_inf_selection")
        nonfinite_weights = pattern == "all_nan"
        if ids_in_range and not degenerate_pattern:
            expected_weights = output_scores.gather(1, actual_ids_i64)
            if need_renorm:
                expected_weights = expected_weights / expected_weights.sum(
                    dim=-1, keepdim=True
                )
            expected_weights = expected_weights * scale
            weight_err = checkAllclose(
                actual_weights,
                expected_weights,
                rtol=2e-3,
                atol=2e-3,
                tol_err_ratio=0.0,
                printLog=False,
            )

            selected_scores = eligible_scores.gather(1, actual_ids_i64)
            max_selection_gap = (
                (kth - selected_scores.min(dim=-1).values).clamp_min(0).max()
            )
            inversions = selected_scores[:, 1:] > selected_scores[:, :-1] + 1e-6
            order_inversion_ratio = inversions.float().mean().item()
        elif ids_in_range:
            weight_err = 0
            max_selection_gap = torch.tensor(0.0)
            order_inversion_ratio = 0.0

        max_selection_gap_f = float(max_selection_gap)
        legacy_ids_match = True
        legacy_weights_match = True
        if legacy_ids is not None:
            actual_ids_sorted, actual_perm = actual_ids.sort(dim=-1)
            legacy_ids_sorted, legacy_perm = legacy_ids.sort(dim=-1)
            legacy_ids_match = torch.equal(actual_ids_sorted, legacy_ids_sorted)
            if legacy_ids_match:
                actual_weights_sorted = actual_weights.gather(1, actual_perm)
                legacy_weights_sorted = legacy_weights.gather(1, legacy_perm)
                legacy_weights_match = torch.allclose(
                    actual_weights_sorted,
                    legacy_weights_sorted,
                    rtol=1e-6,
                    atol=2e-7,
                    equal_nan=True,
                )
        order_compatible = (
            ret["expected route"] != "register" or order_inversion_ratio == 0
        )
        passed = (
            ids_in_range
            and ids_unique
            and (finite or nonfinite_weights)
            and weight_guard
            and id_guard
            and weight_err == 0
            and max_selection_gap_f <= 2e-3
            and legacy_tie_compatible
            and legacy_ids_match
            and legacy_weights_match
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = ops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = 0 if passed else 1
        ret[f"{name} weight err"] = weight_err
        ret[f"{name} max selection gap"] = max_selection_gap_f
        ret[f"{name} order inversion ratio"] = order_inversion_ratio
        ret[f"{name} order compatible"] = order_compatible
        ret[f"{name} legacy tie compatible"] = legacy_tie_compatible
        ret[f"{name} legacy ids match"] = legacy_ids_match
        ret[f"{name} legacy weights match"] = legacy_weights_match
        ret[f"{name} guards"] = weight_guard and id_guard
        ret[f"{name} unique ids"] = ids_unique

    return ret


def _boundary_cases() -> list[dict]:
    cases: list[dict] = []

    def add(
        case,
        *,
        mode="biased",
        token=2,
        expert=256,
        topk=8,
        group=1,
        topk_group=1,
        dtype=dtypes.bf16,
        layout="contiguous",
        bias_layout="contiguous",
        output_pad=1,
        need_renorm=True,
        scale=2.5,
        pattern="random",
    ):
        cases.append(
            {
                "case": case,
                "mode": mode,
                "token": token,
                "expert": expert,
                "topk": topk,
                "group": group,
                "topk_group": topk_group,
                "dtype": dtype,
                "layout": layout,
                "bias_layout": bias_layout,
                "output_pad": output_pad,
                "need_renorm": need_renorm,
                "scale": scale,
                "pattern": pattern,
            }
        )

    # Exact GLM-5.2 route plus token/layout boundaries seen by serving.
    for token, layout in (
        (1, "contiguous"),
        (2, "even_stride"),
        (65, "odd_stride"),
        (2, "even_stride_odd_offset"),
        (4096, "contiguous"),
    ):
        add(f"glm52_t{token}_{layout}", token=token, layout=layout)
    add("glm52_bias_odd_offset", bias_layout="odd_offset")
    add("glm52_wide_output_stride", output_pad=7)
    add("glm52_no_renorm", need_renorm=False)

    # Every compile-time expert instantiation and both sides of its dispatch gate.
    for expert in REG_EXPERTS:
        add(f"reg_expert_{expert}", expert=expert)
    for expert in (64, 127, 129, 320, 2112):
        add(f"fallback_expert_{expert}", expert=expert)

    # Register-path top-k lower/upper boundaries and immediate fallback neighbors.
    for topk in (1, 3, 4, 32, 33):
        add(f"topk_boundary_{topk}", topk=topk)

    # All score transforms and input dtypes accepted by the kernel.
    for mode, dtype in itertools.product(
        ("biased", "sigmoid", "softmax"),
        (dtypes.fp32, dtypes.fp16, dtypes.bf16),
    ):
        add(f"mode_{mode}_{str(dtype).split('.')[-1]}", mode=mode, dtype=dtype)

    # All-groups-selected reaches the register path; filtering must stay on LDS.
    for group, topk_group in ((2, 2), (8, 8), (8, 4)):
        add(
            f"groups_{group}_selected_{topk_group}",
            group=group,
            topk_group=topk_group,
        )

    # Candidate staging boundary, fallback, ties, and sigmoid saturation.
    add(
        "pivot_candidates_64",
        mode="softmax",
        token=1,
        expert=2048,
        topk=4,
        dtype=dtypes.fp32,
        pattern="pivot64",
        scale=1.0,
    )
    add(
        "pivot_candidates_65",
        mode="softmax",
        token=1,
        expert=2048,
        topk=4,
        dtype=dtypes.fp32,
        pattern="pivot65",
        scale=1.0,
    )
    for expert in REG_EXPERTS:
        add(f"all_equal_expert_{expert}", expert=expert, pattern="all_equal")
        add(
            f"sixteen_way_tie_expert_{expert}", expert=expert, pattern="sixteen_way_tie"
        )
    add("tie_at_cutoff", pattern="cutoff_tie")
    add("saturated_sigmoid", pattern="saturated")
    add("all_nan", pattern="all_nan")
    add("all_neg_inf_selection", pattern="all_neg_inf_selection")
    return cases


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "register grouped top-k is unsupported on %s; skipping", get_gfx()
        )
        return

    rows = [test_moe_topk_reg_boundary(**case) for case in _boundary_cases()]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "moe_topk_reg_boundary summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    failed = df.loc[df["aiter err"] != 0, "case"].tolist()
    assert not failed, f"boundary failures: {failed}"


if __name__ == "__main__":
    main()
