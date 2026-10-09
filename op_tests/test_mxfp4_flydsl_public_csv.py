# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Execute actual tuned CSV rows through public fused_moe with explicit A8W4."""

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
    )
    return FmoeTuner.run_torch_moe_stage2(
        ref1,
        data["w1_qt"],
        data["w2_qt"],
        data["topk_weights"],
        data["topk_ids"],
        a2_scale=None,
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


def decode_valid_scales(
    payload: torch.Tensor,
    scales: torch.Tensor,
    inter_dim: int,
    block_m: int,
    sorted_ids: torch.Tensor,
    valid_ids: torch.Tensor,
    token: int,
) -> torch.Tensor:
    """Check only E8M0 bytes actually consumed for live sorted intermediate rows."""
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import kas_per_chunk_dw_for

    ids = sorted_ids[: int(valid_ids[0].item())].long()
    valid = (ids & 0xFFFFFF) < token
    rows = torch.arange(ids.numel(), device=ids.device)[valid, None]
    groups = torch.arange(inter_dim // 32, device=ids.device)[None, :]
    chunk_bytes = kas_per_chunk_dw_for(inter_dim) * 4
    if block_m == 16:
        offsets = (
            (rows // 16) * chunk_bytes
            + (groups // 8) * 256
            + (groups % 4) * 64
            + (rows % 16) * 4
            + ((groups // 4) % 2) * 2
        )
    else:
        # Regular chunks pair 16-row halves in adjacent bytes, and groups are
        # 256-byte tiles. This is the existing shared writer/reader swizzle.
        offsets = (
            (rows // 32) * chunk_bytes
            + (groups // 8) * 256
            + (groups % 4) * 64
            + (rows % 16) * 4
            + ((groups // 4) % 2) * 2
            + (rows // 16) % 2
        )
    active_scales = scales.view(torch.uint8).flatten()[offsets].view(dtypes.fp8_e8m0)
    decoded = payload[: ids.numel()][valid].float() * fp4_utils.e8m0_to_f32(
        active_scales
    ).repeat_interleave(32, 1)
    assert torch.isfinite(decoded).all(), "non-finite decoded live intermediate"
    return active_scales


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
) -> dict[str, Any]:
    expected_pair = (kernel1, kernel2) if expected_pair is None else expected_pair
    g1 = _parse_mxfp4_g1_kname(kernel1)
    g2 = parse_flydsl_v2_gemm2_kernel(kernel2)
    if (g1["a_dtype"], g1["out_dtype"]) != ("fp8", "fp8") or g2 is None:
        return {
            "status": "mode_skip",
            "failure_reason": "row is not an A8/FP8 coupled pair",
        }
    data = Mxfp4FlydslTuner._prepare_case(
        token, model_dim, inter_dim, expert, topk, dtypes.bf16, a_dtype="fp8"
    )
    limit_env = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT")
    limit = (
        float(limit_env)
        if limit_env not in (None, "")
        else (7.0 if activation == ActivationType.Swiglu else None)
    )
    bias1 = (
        torch.randn((expert, 2 * inter_dim), device="cuda", dtype=torch.float32)
        if stage1_bias else None
    )
    reference = run_torch(data, topk, activation, limit, bias1=bias1)
    output = torch.empty((token, model_dim), device="cuda", dtype=dtypes.bf16)
    original1, original2 = fm._mxfp4_a4w4_stage1_fw, fm._mxfp4_a4w4_stage2_fw
    calls, intermediate = [], {}

    @functools.wraps(original1)
    def observe1(*args: Any, **kwargs: Any) -> Any:
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
        )
        if payload.dtype == dtypes.fp8:
            active_scales = decode_valid_scales(
                payload, scales, inter_dim, kwargs["block_m"], args[3], args[5], token
            )
            intermediate.update(
                live_scale_min=int(active_scales.view(torch.uint8).min().item()),
                live_scale_max=int(active_scales.view(torch.uint8).max().item()),
            )
        return result

    @functools.wraps(original2)
    def observe2(*args: Any, **kwargs: Any) -> Any:
        calls.append(("g2", kwargs["kernelName2"]))
        intermediate.update(
            reader_payload_dtype=str(args[0].dtype),
            reader_scale_dtype=str(kwargs["a2_scale"].dtype),
        )
        return original2(*args, **kwargs)

    def public() -> torch.Tensor:
        return fm.fused_moe(
            data["input"],
            data["w1_a16"],
            data["w2_a16"],
            data["topk_weights"],
            data["topk_ids"],
            activation=activation,
            quant_type=QuantType.per_1x32,
            quant_type_a=QuantType.per_1x32,
            quant_dtype_a=dtypes.fp8,
            quant_dtype_a2=dtypes.fp8,
            dtype=dtypes.bf16,
            doweight_stage1=False,
            w1_scale=data["w1s_a16"],
            w2_scale=data["w2s_a16"],
            gate_mode="interleave",
            beta=4.0 if activation == ActivationType.Situv2 else None,
            linear_beta=25.0 if activation == ActivationType.Situv2 else None,
            swiglu_limit=limit,
            output=output,
            bias1=bias1,
        )

    ret = {
        "gfx": get_gfx(),
        "lookup_token": fm.get_padded_M(token),
        "lookup_csv_token": lookup_csv_token,
    }
    with patch.object(fm, "_mxfp4_a4w4_stage1_fw", observe1), patch.object(
        fm, "_mxfp4_a4w4_stage2_fw", observe2
    ):
        fm.get_2stage_cfgs.cache_clear()
        for poison in (float("nan"), 11.0):
            output.fill_(poison)
            result = public()
            assert result.data_ptr() == output.data_ptr()
            assert torch.isfinite(result).all()
        actual1 = next((name for stage, name in reversed(calls) if stage == "g1"), "")
        actual2 = next((name for stage, name in reversed(calls) if stage == "g2"), "")
        ret.update(actual_G1=actual1, actual_G2=actual2)
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
        error = cosine_diff_compare(reference, result)
        assert math.isfinite(error) and error <= 0.1
        element_error = checkAllclose(
            reference.float(),
            result.float(),
            rtol=0.1,
            atol=0.05,
            msg="public A8W4 CSV output",
        )
        ret.update(intermediate, element_error=element_error)
    fm.get_2stage_cfgs.cache_clear()
    # Observe once above; timed calls run the original pipeline with no host
    # intermediate inspection or reference/poison operations in its interval.
    candidates = {"public_fused_moe": public}
    flops = token * topk * model_dim * inter_dim * 6
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            data["input"],
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
    parser.add_argument("--stage1-bias", type=int, choices=(0, 1), nargs="+", default=[0])
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
    for index, stage1_bias in itertools.product(indices, args.stage1_bias):
        row = rows[index]
        lookup = select_csv_lookup(rows, row, fm.get_padded_M(int(row["token"])))
        base = {
            "csv_row": index,
            "stage1_bias": stage1_bias,
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
                            *(
                                int(row[col])
                                for col in (
                                    "token",
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
