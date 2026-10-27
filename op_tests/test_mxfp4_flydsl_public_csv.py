# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
r"""Execute actual tuned CSV rows through public fused_moe with explicit A8W4.

Validate a tuned CSV on one GPU (pick it with ``HIP_VISIBLE_DEVICES``)::

    AITER_CONFIG_FMOE=/absolute/path/to/tuned.csv \
      AITER_ONLINE_TUNE=0 AITER_BYPASS_TUNE_CONFIG=0 \
      AITER_FLYDSL_STAGE2_FP8=0 AITER_MXFP4_INTERMEDIATE=0 \
      python op_tests/test_mxfp4_flydsl_public_csv.py \
        --csv /absolute/path/to/tuned.csv --rows 0 1 2 \
        --report /absolute/path/to/public_report.json

The harness passes explicit FP8 activation/intermediate, INTERLEAVE W1, BF16
output, Situv2 4/25 and the resolved Swiglu limit. It observes and delegates
real stage calls, checks live FP8 payload plus E8M0 scales, output identity,
finite values and amplitude-sensitive error. A skipped/fallback pair is
reported separately and returns nonzero. ``--rows`` uses zero-based data rows.
Each report preserves the requested row/pair, the same CSV's primary lookup
row/token/pair and actual observed G1/G2. For M=3 the public key is M=4; a
correct row4 execution is counted as padded token coverage and does not claim
that an unused standalone row3 winner was executed. Large-token tiers follow
the same runtime mapping, including smaller-tier lookup when applicable.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib
import itertools
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    parse_flydsl_v2_gemm2_kernel,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
    FmoeTuner,
    Mxfp4FlydslTuner,
    cosine_diff_compare,
)

fm = importlib.import_module("aiter.fused_moe")
SUPPORTED_GFX = ("gfx950",)
# Shape parameters are supplied by main(), not pytest fixtures.
__test__ = False


def run_torch(
    data: dict[str, Any],
    topk: int,
    activation: ActivationType,
    limit: float | None,
    bias1: torch.Tensor | None = None,
    fp8_intermediate: bool = False,
    beta: float | None = None,
    linear_beta: float | None = None,
) -> torch.Tensor:
    ref1 = FmoeTuner.run_torch_moe_stage1(
        data["a1_qt"],
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        data["a1_scale"],
        data["w1_scale"],
        w1_bias=bias1,
        dtype=dtypes.bf16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        topk=topk,
        swiglu_limit=limit,
        situ_beta=4.0 if beta is None else beta,
        situ_linear_beta=25.0 if linear_beta is None else linear_beta,
    )
    a2_scale = None
    if fp8_intermediate:
        from aiter.ops.quant import per_1x32_f8_scale_f8_quant

        shape = ref1.shape
        ref1, a2_scale = per_1x32_f8_scale_f8_quant(
            ref1.reshape(-1, shape[-1]), scale_type=dtypes.fp8_e8m0
        )
        ref1 = ref1.reshape(shape)
    return FmoeTuner.run_torch_moe_stage2(
        ref1,
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        a2_scale=a2_scale,
        w2_scale=data["w2_scale"],
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
    )


def csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as source:
        return [
            {key.strip(): (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(source)
        ]


def select_csv_lookup(
    rows: list[dict[str, str]], requested: dict[str, Any], lookup_token: int
) -> dict[str, Any] | None:
    """Identify the primary CSV row for the public token key; never override it."""
    fields = (
        "gfx",
        "cu_num",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    )

    def key(row: dict[str, Any]) -> tuple[str, ...]:
        values = []
        for field in fields:
            value = str(row.get(field, "")).strip()
            if field in ("cu_num", "model_dim", "inter_dim", "expert", "topk"):
                value = str(int(float(value)))
            elif field in ("use_g1u1", "doweight_stage1"):
                value = {"true": "1", "false": "0"}.get(value.lower(), value)
            values.append(value)
        return tuple(values)

    tiers = [lookup_token]
    if lookup_token > fm._PADDED_M_TIERS[0]:
        index = (
            fm._PADDED_M_TIERS.index(lookup_token)
            if lookup_token in fm._PADDED_M_TIERS
            else -1
        )
        tiers.extend(reversed(fm._PADDED_M_TIERS[:index]))
    requested_key = key(requested)
    for token in tiers:
        for index, row in enumerate(rows):
            if (
                row.get("_tag") != "flydsl_fallback"
                and int(float(row["token"])) == token
                and key(row) == requested_key
            ):
                return {
                    "csv_row": index,
                    "token": token,
                    "kernelName1": row["kernelName1"],
                    "kernelName2": row["kernelName2"],
                    "source_tag": row.get("_tag", ""),
                }
    return None


def scale_byte_offsets(rows, groups, inter_dim: int, *, native: bool):
    """Map logical rows/groups to the shared producer/consumer byte layout."""
    chunk_bytes = ((inter_dim // 32 + 7) // 8) * 256
    offsets = (
        (rows // (16 if native else 32)) * chunk_bytes
        + (groups // 8) * 256
        + (groups % 4) * 64
        + (rows % 16) * 4
        + ((groups // 4) % 2) * 2
    )
    return offsets if native else offsets + (rows // 16) % 2


def decode_valid_scales(
    payload: torch.Tensor,
    scales: torch.Tensor,
    inter_dim: int,
    block_m: int,
    sorted_ids: torch.Tensor,
    valid_ids: torch.Tensor,
    token: int,
    native_scale_layout: bool | None = None,
) -> torch.Tensor:
    """Check only E8M0 bytes actually consumed for live sorted intermediate rows."""
    ids = sorted_ids[: int(valid_ids[0].item())].long()
    valid = (ids & 0xFFFFFF) < token
    rows = torch.arange(ids.numel(), device=ids.device)[valid, None]
    groups = torch.arange(inter_dim // 32, device=ids.device)[None, :]
    if native_scale_layout is None:
        native_scale_layout = block_m == 16
    offsets = scale_byte_offsets(rows, groups, inter_dim, native=native_scale_layout)
    active_scales = scales.view(torch.uint8).flatten()[offsets].view(dtypes.fp8_e8m0)
    decoded = payload[: ids.numel()][valid].float() * fp4_utils.e8m0_to_f32(
        active_scales
    ).repeat_interleave(32, 1)
    assert torch.isfinite(decoded).all(), "non-finite decoded live intermediate"
    return active_scales


def inspect_regular_scales(
    payload, scales, sorted_ids, valid_ids, token, topk, inter_dim, block_m, scale_byte
):
    """Check selected bytes separately from the reader's full 4-byte loads."""
    sorted_rows = int(valid_ids[0].item())
    ids = sorted_ids[:sorted_rows].long()
    live = ((ids & 0xFFFFFF) < token) & ((ids >> 24) < topk)
    rows = torch.arange(sorted_rows, device=ids.device)[:, None]
    groups = torch.arange(inter_dim // 32, device=ids.device)[None, :]
    offsets = scale_byte_offsets(rows, groups, inter_dim, native=False)
    raw = scales.view(torch.uint8).flatten()
    physical_cols = ((inter_dim // 32 + 7) // 8) * 8
    assert scales.is_contiguous() and scales.shape[1] == physical_cols
    assert scales.stride() == (physical_cols, 1)
    assert scales.shape[0] >= (sorted_ids.numel() + 31) // 32 * 32
    # All active sorted blocks include padding lanes. They load whole words;
    # only live routes and valid K tiles contribute selected bytes to output.
    words = (offsets // 4 * 4).unique()
    assert int(words.min()) >= 0 and int(words.max()) + 3 < raw.numel()
    selector = ((groups // 4) % 2) * 2 + (rows // 16) % 2
    assert torch.equal(offsets, offsets // 4 * 4 + selector)
    if scale_byte == 255:
        assert (raw[offsets[live]] != 255).all(), "producer omitted live scale bytes"
    padded_groups = torch.arange(inter_dim // 32, physical_cols, device=ids.device)[
        None, :
    ]
    padding_offsets = scale_byte_offsets(rows, padded_groups, inter_dim, native=False)
    trailing = torch.empty(0, device=ids.device, dtype=torch.long)
    if sorted_rows % 32:
        tail_rows = torch.arange(
            sorted_rows, (sorted_rows + 31) // 32 * 32, device=ids.device
        )[:, None]
        trailing = scale_byte_offsets(tail_rows, groups, inter_dim, native=False)
    if scale_byte == 255:
        assert (raw[padding_offsets] == 255).all(), "unexpected padded K-pack write"
        assert (raw[trailing] == 255).all(), "unexpected trailing row-half write"
    active_scales = raw[offsets[live]].view(dtypes.fp8_e8m0)
    decoded = payload[:sorted_rows][live].float() * fp4_utils.e8m0_to_f32(
        active_scales
    ).repeat_interleave(32, 1)
    assert torch.isfinite(decoded).all()
    return (
        decoded,
        active_scales,
        {
            "scale_shape": tuple(scales.shape),
            "scale_stride": tuple(scales.stride()),
            "sorted_rows": sorted_rows,
            "active_blocks": sorted_rows // block_m,
            "last_row_half": ((sorted_rows // 16) - 1) % 2,
            "live_routes": int(live.sum()),
            "live_scale_bytes": offsets[live].numel(),
            "writer_overwritten_bytes": (
                int((raw != scale_byte).sum()) if scale_byte is not None else None
            ),
            "reader_word_min": int(words.min()),
            "reader_word_end": int(words.max()) + 3,
            "allocation_bytes": raw.numel(),
            "padded_k_bytes": padding_offsets.numel(),
            "unwritten_row_half_bytes": trailing.numel(),
            "selector_checked": True,
        },
    )


@benchmark()
def test_public_a8w4_csv(
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    activation: ActivationType,
    kernel1: str,
    kernel2: str,
    expected_pair: tuple[str, str] | None = None,
    lookup_csv_token: int | None = None,
    stage1_bias: int = 0,
    routing: str = "random",
    block_amplitude: float = 1.0,
    scale_byte: int | None = None,
    input_mode: str = "raw",
    activation_params: str = "default",
    intermediate_precision: str = "fp8",
    stage2_fp8: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    expected_pair = (kernel1, kernel2) if expected_pair is None else expected_pair
    g1 = _parse_mxfp4_g1_kname(kernel1)
    g2 = parse_flydsl_v2_gemm2_kernel(kernel2)
    is_opus = fm._opus_a8w4.is_opus_a8w4_stage2_kernel(kernel2)
    if (g1["a_dtype"], g1["out_dtype"]) != ("fp8", "fp8") or (
        g2 is None and not is_opus
    ):
        return {
            "status": "mode_skip",
            "failure_reason": "row is not an A8/FP8 coupled pair",
        }
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtypes.bf16, a_dtype="fp8"
    )
    if routing != "random":
        assert expert > topk and token > g1["BM"]
        data["topk_ids"] = (
            torch.arange(topk, dtype=dtypes.i32, device="cuda")
            .expand(token, topk)
            .contiguous()
        )
        if routing == "fixed-tail":
            data["topk_ids"][0, 0] = topk
        data["topk_weights"] = (
            torch.arange(1, topk + 1, dtype=dtypes.fp32, device="cuda")
            .expand(token, topk)
            .contiguous()
        )
        data["topk_weights"] /= topk * (topk + 1) / 2
        route_counts = torch.bincount(
            data["topk_ids"].flatten().long(), minlength=expert
        )
        expected_counts = [token] * topk
        if routing == "fixed-tail":
            expected_counts[0] -= 1
            expected_counts.append(1)
        assert route_counts[: len(expected_counts)].tolist() == expected_counts
        assert int(route_counts[len(expected_counts) :].sum()) == 0
    if block_amplitude != 1.0:
        from aiter.ops.quant import per_1x32_mx_quant_hip
        from aiter.ops.shuffle import shuffle_weight

        data["input"][g1["BM"] :] *= block_amplitude
        zero_columns = torch.cat((torch.arange(32), inter_dim + torch.arange(32)))
        data["w1_qt"].view(torch.uint8)[:, zero_columns] = 0
        data["w1_a16"] = shuffle_weight(
            data["w1_qt"], (16, 16), is_guinterleave=True, gate_up=True
        )
        data["a1_qt"], data["a1_scale"] = per_1x32_mx_quant_hip(
            data["input"],
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
        )
    limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
    limit = (
        float(limit_env)
        if limit_env not in (None, "")
        else (7.0 if activation == ActivationType.Swiglu else None)
    )
    beta = linear_beta = None
    if activation_params != "default":
        values = tuple(float(value) for value in activation_params.split(","))
        if activation == ActivationType.Situv2:
            assert len(values) == 2
            beta, linear_beta = values
        else:
            assert len(values) == 1
            limit = values[0]
    assert input_mode in ("raw", "prequant")
    if input_mode == "prequant":
        assert not g1["inline_quant"], "prequantized FP8 requires non-inline G1"
    public_input = data["input"] if input_mode == "raw" else data["a1_qt"]
    public_input_scale = None if input_mode == "raw" else data["a1_scale"]
    if public_input_scale is not None:
        assert public_input_scale.shape == (token, model_dim // 32)
    bias1 = (
        torch.randn((expert, 2 * inter_dim), device="cuda", dtype=torch.float32)
        if stage1_bias
        else None
    )
    reference = run_torch(
        data,
        topk,
        activation,
        limit,
        bias1=bias1,
        fp8_intermediate=is_opus,
        beta=beta,
        linear_beta=linear_beta,
    )
    output = torch.empty((token, model_dim), device="cuda", dtype=dtypes.bf16)
    original1 = fm._mxfp4_a4w4_stage1_fw
    original2 = (
        fm._opus_a8w4.opus_a8w4_stage2_wrapper if is_opus else fm._mxfp4_a4w4_stage2_fw
    )
    calls, intermediate, sort_calls = [], {}, []
    control_intermediate = None
    current_scale_byte = None
    launcher = importlib.import_module("aiter.ops.flydsl.mxfp4_gemm1_kernels")
    original_launch = launcher.flydsl_mxfp4_gemm1
    original_generated_sort = aiter.mxfp4_moe_sort
    original_opus_sort = aiter.moe_sorting_opus_fwd
    original_scale_sort = fm.mxfp4_moe_sort_fwd
    original_decode = fm._opus_a8w4._opus_moe_stage2_a8w4_decode_fwd_raw
    original_reduce = fm._opus_a8w4.opus_moe_stage2_reduce_token_slot_route_output_fwd
    original_cfg = fm.get_2stage_cfgs
    metadata_seen, workspace_calls, scale_sort_calls = [], [], []

    def observe_cfg(*args, **kwargs):
        result = original_cfg(*args, **kwargs)
        metadata_seen.append(result)
        return result

    observe_cfg.cache_clear = original_cfg.cache_clear

    def observe_scale_sort(*args, **kwargs):
        result = original_scale_sort(*args, **kwargs)
        scale_sort_calls.append((args[0], result))
        return result

    def observe_decode(*args, **kwargs):
        workspace = args[8]
        workspace_calls.append(
            {
                "stage": "g2",
                "dtype": str(workspace.dtype),
                "shape": tuple(workspace.shape),
                "kid": int(args[12]),
            }
        )
        return original_decode(*args, **kwargs)

    def observe_reduce(route_out, *args, **kwargs):
        workspace_calls.append(
            {
                "stage": "reduce",
                "dtype": str(route_out.dtype),
                "shape": tuple(route_out.shape),
            }
        )
        return original_reduce(route_out, *args, **kwargs)

    def observe_launch(*args, **kwargs):
        if current_scale_byte is not None:
            kwargs["inter_sorted_shuffled_scale"].view(torch.uint8).fill_(
                current_scale_byte
            )
        return original_launch(*args, **kwargs)

    def observe_generated_sort(*args, **kwargs):
        sort_calls.append("generated")
        return original_generated_sort(*args, **kwargs)

    def observe_opus_sort(*args, **kwargs):
        sort_calls.append("opus")
        return original_opus_sort(*args, **kwargs)

    @functools.wraps(original1)
    def observe1(*args: Any, **kwargs: Any) -> Any:
        nonlocal control_intermediate
        result = original1(*args, **kwargs)
        calls.append(("g1", kwargs["kernelName1"]))
        payload, scales = result
        intermediate.update(
            payload_dtype=str(payload.dtype),
            scale_dtype=str(scales.dtype),
            scale_bytes=scales.numel(),
            input_dtype=str(args[0].dtype),
            inline=_parse_mxfp4_g1_kname(kwargs["kernelName1"])["inline_quant"],
            interleave=kwargs["interleave"],
            prequant_has_scales=kwargs.get("a1_scale") is not None,
            stage1_has_bias=kwargs.get("bias1") is not None,
            native_scale_layout=kwargs.get("native_scale_layout", g1["BM"] == 16),
            scale_shape=tuple(scales.shape),
            scale_stride=tuple(scales.stride()),
            input_payload_preserved=(args[0].data_ptr() == public_input.data_ptr()),
            actual_beta=kwargs.get("situ_beta"),
            actual_linear_beta=kwargs.get("situ_linear_beta"),
            actual_limit=kwargs.get("swiglu_limit"),
        )
        if payload.dtype == dtypes.fp8:
            active_scales = decode_valid_scales(
                payload,
                scales,
                inter_dim,
                kwargs["block_m"],
                args[3],
                args[5],
                token,
                native_scale_layout=kwargs.get("native_scale_layout"),
            )
            intermediate.update(
                live_scale_min=int(active_scales.view(torch.uint8).min().item()),
                live_scale_max=int(active_scales.view(torch.uint8).max().item()),
            )
        if is_opus and scale_byte is not None:
            decoded, active_scales, contract = inspect_regular_scales(
                payload,
                scales,
                args[3],
                args[5],
                token,
                topk,
                inter_dim,
                kwargs["block_m"],
                current_scale_byte,
            )
            intermediate.update(contract)
            ids = args[3][: int(args[5][0].item())].long()
            live = ((ids & 0xFFFFFF) < token) & ((ids >> 24) < topk)
            route_ids = (ids[live] & 0xFFFFFF) * topk + (ids[live] >> 24)
            route_order = route_ids.argsort()
            assert torch.equal(
                route_ids[route_order],
                torch.arange(token * topk, device=route_ids.device),
            ), "sort omitted or duplicated a live route"
            live_payload = payload[: ids.numel()][live][route_order].view(torch.uint8)
            routed_scales = active_scales[route_order].view(torch.uint8)
            if current_scale_byte is None:
                if control_intermediate is not None:
                    changed = (live_payload != control_intermediate[0]).nonzero()
                    assert not changed.numel(), (
                        f"normal routed payload changed at {changed[:8].tolist()}, "
                        f"bytes={live_payload[changed[:, 0], changed[:, 1]][:8].tolist()} vs "
                        f"{control_intermediate[0][changed[:, 0], changed[:, 1]][:8].tolist()}"
                    )
                control_intermediate = (live_payload.clone(), routed_scales.clone())
            else:
                changed = (live_payload != control_intermediate[0]).nonzero()
                assert torch.equal(live_payload, control_intermediate[0]), (
                    f"poison changed routed FP8 payload at {changed[:8].tolist()}, "
                    f"bytes={live_payload[changed[:, 0], changed[:, 1]][:8].tolist()} vs "
                    f"{control_intermediate[0][changed[:, 0], changed[:, 1]][:8].tolist()}"
                )
                assert torch.equal(
                    routed_scales, control_intermediate[1]
                ), "poison changed routed scale bytes"
                intermediate["normal_poison_intermediate_equal"] = True
            if g1["BM"] == 16:
                native_q, native_s = original1(
                    *args, **dict(kwargs, native_scale_layout=True)
                )
                native_scale = decode_valid_scales(
                    native_q,
                    native_s,
                    inter_dim,
                    16,
                    args[3],
                    args[5],
                    token,
                    native_scale_layout=True,
                )
                native_values = native_q[: ids.numel()][
                    live
                ].float() * fp4_utils.e8m0_to_f32(native_scale).repeat_interleave(32, 1)
                assert torch.equal(
                    native_values, decoded
                ), "native/regular live values differ"
                assert torch.equal(
                    native_scale.view(torch.uint8), active_scales.view(torch.uint8)
                ), "native/regular live scale bytes differ"
                intermediate["native_regular_equivalent"] = True
            if block_amplitude != 1.0:
                assert torch.equal(
                    decoded[:, :32], torch.zeros_like(decoded[:, :32])
                ), "expected zero intermediate group"
                live_indices = torch.where(live)[0]
                first = active_scales[live_indices // 16 == 0, 1:].view(torch.uint8)
                second = active_scales[live_indices // 16 == 1, 1:].view(torch.uint8)
                assert first.numel() and second.numel()
                shared_rows = min(first.shape[0], second.shape[0])
                assert not torch.equal(
                    first[:shared_rows], second[:shared_rows]
                ), "first BM16 blocks need distinct scales"
                intermediate["distinct_bm16_block_scales"] = True
        return result

    @functools.wraps(original2)
    def observe2(*args: Any, **kwargs: Any) -> Any:
        calls.append(("g2", kwargs["kernelName" if is_opus else "kernelName2"]))
        intermediate.update(
            reader_payload_dtype=str(args[0].dtype),
            reader_scale_dtype=str(kwargs["a2_scale"].dtype),
            stage2_weights_present=kwargs.get("sorted_weights") is not None,
            **({"opus_kid": kwargs["launch"].kernel_id} if is_opus else {}),
        )
        return original2(*args, **kwargs)

    def public() -> torch.Tensor:
        return fm.fused_moe(
            public_input,
            data["w1_a16"],
            data["w2_a16"],
            data["topk_weights"],
            data["topk_ids"],
            activation=activation,
            quant_type=QuantType.per_1x32,
            quant_type_a=QuantType.per_1x32,
            quant_dtype_a=dtypes.fp8,
            quant_dtype_a2=None if intermediate_precision == "none" else dtypes.fp8,
            dtype=dtypes.bf16,
            doweight_stage1=False,
            w1_scale=data["w1s_a16"],
            w2_scale=data["w2s_a16"],
            a1_scale=public_input_scale,
            gate_mode="interleave",
            beta=beta,
            linear_beta=linear_beta,
            swiglu_limit=limit,
            output=output,
            bias1=bias1,
        )

    ret = {
        "gfx": get_gfx(),
        "lookup_token": fm.get_padded_M(token),
        "lookup_csv_token": lookup_csv_token,
    }
    if routing != "random":
        ret["route_counts"] = expected_counts
    uses_generated = g1["BM"] == 16 and token * topk < expert
    if uses_generated and fm._MOE_SORT_BACKEND == "auto":
        zero_init = (
            not fm._opus_a8w4.parse_stage2_config(
                {"kernelName2": kernel2}, g1["BM"]
            ).route_out
            if is_opus
            else g2["epilog"] == "atomic"
        )
        supported = fm._mxfp4_aux_instance_supported(
            expert, topk, model_dim, g1["BM"], zero_init
        )
        assert supported, "actual generated auxiliary instance is unavailable"
        ret["compiled_aux_supported"] = supported
        module = importlib.import_module("aiter.jit.core").get_module(
            "module_moe_mxfp4_aux"
        )
        ret["compiled_aux_module"] = module.__file__
        ret["compiled_aux_sha256"] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        ret["compiled_aux_key"] = (
            f"aux_{'sortzi' if zero_init else 'sortonly'}_NE{expert}"
            f"_TOPK{topk}_MB{g1['BM']}_H{model_dim}"
        )
    with ExitStack() as stack:
        stack.enter_context(patch.object(fm, "_mxfp4_a4w4_stage1_fw", observe1))
        stack.enter_context(
            patch.object(
                fm._opus_a8w4 if is_opus else fm,
                "opus_a8w4_stage2_wrapper" if is_opus else "_mxfp4_a4w4_stage2_fw",
                observe2,
            )
        )
        stack.enter_context(
            patch.object(launcher, "flydsl_mxfp4_gemm1", observe_launch)
        )
        stack.enter_context(
            patch.object(aiter, "mxfp4_moe_sort", observe_generated_sort)
        )
        stack.enter_context(
            patch.object(aiter, "moe_sorting_opus_fwd", observe_opus_sort)
        )
        stack.enter_context(patch.object(fm, "mxfp4_moe_sort_fwd", observe_scale_sort))
        stack.enter_context(patch.object(fm, "get_2stage_cfgs", observe_cfg))
        if is_opus:
            stack.enter_context(
                patch.object(
                    fm._opus_a8w4,
                    "_opus_moe_stage2_a8w4_decode_fwd_raw",
                    observe_decode,
                )
            )
            stack.enter_context(
                patch.object(
                    fm._opus_a8w4,
                    "opus_moe_stage2_reduce_token_slot_route_output_fwd",
                    observe_reduce,
                )
            )
        fm.get_2stage_cfgs.cache_clear()
        control = public().clone() if scale_byte is not None else None
        if control is not None:
            normal_repeat = public()
            repeat_error = cosine_diff_compare(control, normal_repeat, printLog=False)
            repeat_mismatch = checkAllclose(
                control.float(),
                normal_repeat.float(),
                rtol=0.1,
                atol=0.05,
                msg="normal initialization repeat",
            )
            assert math.isfinite(repeat_error) and repeat_error <= 0.1
            assert math.isfinite(repeat_mismatch) and repeat_mismatch <= 0.05
            ret["normal_repeat_error"] = repeat_error
            ret["normal_repeat_mismatch"] = repeat_mismatch
        current_scale_byte = scale_byte
        env_sequence = stage2_fp8 or (int(fm._opus_stage2_fp8_enabled()),)
        route_launch = (
            fm._opus_a8w4.parse_stage2_config({"kernelName2": kernel2}, g1["BM"])
            if is_opus
            else None
        )
        env_records = []
        poison_calls = itertools.product(env_sequence, (float("nan"), 11.0))
        for enabled, poison in poison_calls:
            workspace_begin = len(workspace_calls)
            output.fill_(poison)
            with patch.dict(os.environ, {"AITER_OPUS_STAGE2_FP8": str(enabled)}):
                result = public()
            assert result.data_ptr() == output.data_ptr()
            assert torch.isfinite(result).all()
            if control is not None:
                repeat_error = checkAllclose(
                    control.float(),
                    result.float(),
                    rtol=0.1,
                    atol=0.05,
                    msg="scale initialization control",
                )
                assert math.isfinite(repeat_error) and repeat_error <= 0.05
                control_error = cosine_diff_compare(control, result, printLog=False)
                assert math.isfinite(control_error) and control_error <= 0.1
                ret["normal_poison_error"] = control_error
                ret["normal_poison_mismatch"] = repeat_error
            if is_opus:
                error = cosine_diff_compare(reference, result)
                assert math.isfinite(error) and error <= 0.1
                mismatch = checkAllclose(
                    reference.float(),
                    result.float(),
                    rtol=0.1,
                    atol=0.05,
                    tol_err_ratio=0.05,
                    msg="MXMOE -> OPUS output poison",
                )
                assert math.isfinite(mismatch) and mismatch <= 0.05
                actual_workspace = workspace_calls[workspace_begin:]
                expected_dtype = str(
                    torch.uint8
                    if route_launch.route_out_fp8 and enabled
                    else dtypes.bf16
                )
                assert actual_workspace and all(
                    call["dtype"] == expected_dtype for call in actual_workspace
                )
                assert actual_workspace[0]["kid"] == route_launch.kernel_id
                assert [call["stage"] for call in actual_workspace] == (
                    ["g2", "reduce"] if route_launch.route_out else ["g2"]
                )
                env_records.append(
                    {
                        "enabled": enabled,
                        "output_poison": str(poison),
                        "workspace": actual_workspace,
                        "error": error,
                        "mismatch": mismatch,
                    }
                )
        actual1 = next((name for stage, name in reversed(calls) if stage == "g1"), "")
        actual2 = next((name for stage, name in reversed(calls) if stage == "g2"), "")
        ret.update(actual_G1=actual1, actual_G2=actual2)
        ret["sort_backends"] = sorted(set(sort_calls))
        assert (
            len(ret["sort_backends"]) == 1
        ), "public sort backend changed between calls"
        if fm._MOE_SORT_BACKEND == "auto":
            assert ret["sort_backends"] == ["generated" if uses_generated else "opus"]
        expected = (
            [("g1", expected_pair[0]), ("g2", expected_pair[1])]
            if expected_pair
            else []
        )
        if not expected_pair or calls[-2:] != expected:
            ret.update(
                status="fallback",
                actual_calls=json.dumps(calls),
                failure_reason="public call did not execute the primary row selected by the token lookup",
            )
            return ret
        assert (
            intermediate["payload_dtype"]
            == intermediate["reader_payload_dtype"]
            == str(dtypes.fp8)
        )
        assert (
            intermediate["scale_dtype"]
            == intermediate["reader_scale_dtype"]
            == str(dtypes.fp8_e8m0)
        )
        assert intermediate["interleave"] is True
        assert intermediate["stage1_has_bias"] == bool(stage1_bias)
        assert intermediate["inline"] == (
            intermediate["input_dtype"] == str(dtypes.bf16)
        )
        assert intermediate["inline"] or intermediate["prequant_has_scales"]
        if input_mode == "prequant":
            assert intermediate["input_payload_preserved"]
            assert scale_sort_calls and all(
                source is public_input_scale for source, _ in scale_sort_calls
            )
            ret["original_scales_sorted"] = True
        if is_opus:
            assert intermediate["stage2_weights_present"]
            assert metadata_seen and all(
                meta is metadata_seen[0] for meta in metadata_seen[::2]
            )
            ret["metadata_cache_preserved"] = True
            ret["stage2_env_calls"] = env_records
            ret["g1_BM"] = g1["BM"]
            ret["opus_BM"] = route_launch.instance.block_m
            ret["opus_SBM"] = route_launch.sort_block_m
            if activation == ActivationType.Situv2:
                assert intermediate["actual_beta"] == (4.0 if beta is None else beta)
                assert intermediate["actual_linear_beta"] == (
                    25.0 if linear_beta is None else linear_beta
                )
            else:
                assert intermediate["actual_limit"] == limit
        error = cosine_diff_compare(reference, result)
        assert math.isfinite(error) and error <= 0.1
        element_error = checkAllclose(
            reference.float(),
            result.float(),
            rtol=0.1,
            atol=0.05,
            msg="public A8W4 CSV output",
        )
        if is_opus:
            assert math.isfinite(element_error) and element_error <= 0.05
        ret.update(intermediate, element_error=element_error)
    fm.get_2stage_cfgs.cache_clear()
    # Observe once above; timed calls run the original pipeline with no host
    # intermediate inspection or reference/poison operations in its interval.
    candidates = {"public_fused_moe": public}
    flops = token * topk * model_dim * inter_dim * 6
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            public_input,
            data["w1_a16"],
            data["w2_a16"],
            data["w1s_a16"],
            data["w2s_a16"],
            data["topk_ids"],
            data["topk_weights"],
            output,
            *([bias1] if bias1 is not None else []),
        )
    )
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        assert torch.isfinite(result).all()
        err = cosine_diff_compare(reference, result)
        assert math.isfinite(err) and err <= 0.1 and math.isfinite(us) and us > 0
        if is_opus:
            mismatch = checkAllclose(
                reference.float(),
                result.float(),
                rtol=0.1,
                atol=0.05,
                tol_err_ratio=0.05,
                msg="MXMOE -> OPUS public",
            )
            assert math.isfinite(mismatch) and mismatch <= 0.05
            ret[f"{name} mismatch_ratio"] = mismatch
            ret["opus_kid"] = intermediate["opus_kid"]
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    ret.update(
        status=(
            "padded_token_pair_hit"
            if token != (lookup_csv_token or fm.get_padded_M(token))
            else "exact_pair_hit"
        ),
        failure_reason="",
    )
    return ret


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="zero-based CSV data row indices; default all",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--stage1-bias", type=int, choices=(0, 1), nargs="+", default=[0]
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="+",
        default=None,
        help="actual token counts; CSV lookup uses their padded token keys",
    )
    parser.add_argument(
        "--routing",
        choices=("random", "fixed", "fixed-tail"),
        nargs="+",
        default=["random"],
    )
    parser.add_argument("--block-amplitude", type=float, nargs="+", default=[1.0])
    parser.add_argument(
        "--scale-byte", type=int, choices=(0, 255), nargs="+", default=[None]
    )
    parser.add_argument(
        "--input-mode", choices=("raw", "prequant"), nargs="+", default=["raw"]
    )
    parser.add_argument(
        "--activation-params",
        nargs="+",
        default=["default"],
        help="default, a SiLU/Swiglu limit, or Situv2 beta,linear_beta",
    )
    parser.add_argument(
        "--intermediate-precision", choices=("none", "fp8"), nargs="+", default=["fp8"]
    )
    parser.add_argument(
        "--stage2-fp8",
        type=dtypes.str2tuple,
        nargs="+",
        default=[None],
        help="per-call env sequence for one cached pair, e.g. 1,0,1",
    )
    args = parser.parse_args()
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("public A8W4 CSV validation unsupported on %s", get_gfx())
        return
    rows = csv_rows(args.csv)
    indices = args.rows if args.rows is not None else list(range(len(rows)))
    os.environ["AITER_CONFIG_FMOE"] = str(args.csv.resolve())
    fm.AITER_CONFIGS.get_config_file.cache_clear()
    fm.cfg_2stages = None
    fm.get_2stage_cfgs.cache_clear()
    records = []
    for (
        index,
        stage1_bias,
        batch,
        routing,
        amplitude,
        scale_byte,
        input_mode,
        activation_params,
        intermediate_precision,
        stage2_fp8,
    ) in itertools.product(
        indices,
        args.stage1_bias,
        args.batch or [None],
        args.routing,
        args.block_amplitude,
        args.scale_byte,
        args.input_mode,
        args.activation_params,
        args.intermediate_precision,
        args.stage2_fp8,
    ):
        row = rows[index]
        token = int(row["token"]) if batch is None else batch
        lookup = select_csv_lookup(rows, row, fm.get_padded_M(token))
        base = {
            "csv_row": index,
            "stage1_bias": stage1_bias,
            "actual_token": token,
            "routing": routing,
            "block_amplitude": amplitude,
            "scale_byte": scale_byte,
            "input_mode": input_mode,
            "activation_params": activation_params,
            "intermediate_precision": intermediate_precision,
            "stage2_fp8": stage2_fp8,
            "source_tag": row.get("_tag", ""),
            "shape": row,
            "requested_pair": {"G1": row["kernelName1"], "G2": row["kernelName2"]},
            "lookup_csv_row": lookup["csv_row"] if lookup else None,
            "lookup_csv_token": lookup["token"] if lookup else None,
            "lookup_pair": (
                {"G1": lookup["kernelName1"], "G2": lookup["kernelName2"]}
                if lookup
                else None
            ),
        }
        try:
            if row.get("_tag") == "flydsl_fallback" or row["q_dtype_a"] != str(
                dtypes.fp8
            ):
                base.update(
                    status="mode_skip",
                    failure_reason="filtered tag or non-A8 execution row",
                )
            else:
                activation = getattr(ActivationType, row["act_type"].split(".")[-1])
                with torch.device("cuda"):
                    base.update(
                        test_public_a8w4_csv(
                            token,
                            *(
                                int(row[col])
                                for col in (
                                    "model_dim",
                                    "inter_dim",
                                    "expert",
                                    "topk",
                                )
                            ),
                            activation,
                            row["kernelName1"],
                            row["kernelName2"],
                            expected_pair=(
                                (lookup["kernelName1"], lookup["kernelName2"])
                                if lookup
                                else ()
                            ),
                            lookup_csv_token=lookup["token"] if lookup else None,
                            stage1_bias=stage1_bias,
                            routing=routing,
                            block_amplitude=amplitude,
                            scale_byte=scale_byte,
                            input_mode=input_mode,
                            activation_params=activation_params,
                            intermediate_precision=intermediate_precision,
                            stage2_fp8=stage2_fp8,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            base.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        records.append(base)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "csv": str(args.csv.resolve()),
                    "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                    "records": records,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
    aiter.logger.info(
        "public A8W4 CSV summary (markdown):\n%s",
        pd.DataFrame(
            [
                {key: value for key, value in row.items() if key != "shape"}
                for row in records
            ]
        ).to_markdown(index=False),
    )
    if any(
        row["status"] not in ("exact_pair_hit", "padded_token_pair_hit")
        for row in records
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
